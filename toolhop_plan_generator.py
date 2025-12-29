"""
ToolHop Candidate Plan Generator with Validation

This script generates N candidate plans from ToolHop dataset with:
1. Plan generation (with diverse error injection)
2. Static validation
3. LLM judge annotation
4. Comprehensive debugging and validation
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import re
from collections import defaultdict
import anthropic
import openai
from tqdm import tqdm
import random


class ErrorType(Enum):
    """Types of errors to inject into plans"""
    NONE = "none"  # Ground truth plan
    TYPE_MISMATCH = "type_mismatch"
    MISSING_DEPENDENCY = "missing_dependency"
    WRONG_TOOL = "wrong_tool"
    PARAMETER_TYPO = "parameter_typo"
    INEFFICIENT_ORDER = "inefficient_order"
    INCOMPLETE_PLAN = "incomplete_plan"
    UNNECESSARY_STEPS = "unnecessary_steps"
    CIRCULAR_DEPENDENCY = "circular_dependency"


@dataclass
class ToolCall:
    """Represents a single tool call in a plan"""
    step_id: int
    tool_name: str
    parameters: Dict[str, Any]
    output_variable: str  # e.g., "{{0}}", "{{1}}"
    expected_output: Optional[str] = None
    
    def __str__(self):
        params_str = ", ".join([f"{k}={repr(v)}" for k, v in self.parameters.items()])
        return f"Step {self.step_id}: {self.output_variable} = {self.tool_name}({params_str})"


@dataclass
class Plan:
    """Represents a complete plan"""
    steps: List[ToolCall]
    error_type: ErrorType = ErrorType.NONE
    
    def to_dict(self):
        return {
            "steps": [asdict(step) for step in self.steps],
            "error_type": self.error_type.value
        }
    
    def __str__(self):
        return "\n".join([str(step) for step in self.steps])


@dataclass
class JudgeAnnotation:
    """Annotation from LLM judge"""
    quality_score: int  # 0-100
    success_prediction: str  # "yes", "likely_yes", "uncertain", "likely_no", "no"
    reasoning: str
    issues: List[Dict[str, Any]]  # [{"type": "...", "severity": "...", "description": "..."}]
    confidence: float  # 0.0-1.0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class AnnotatedPlan:
    """A plan with its judge annotation"""
    plan: Plan
    annotation: JudgeAnnotation
    query_id: int
    
    def to_dict(self):
        return {
            "query_id": self.query_id,
            "plan": self.plan.to_dict(),
            "annotation": self.annotation.to_dict()
        }


class GroundTruthParser:
    """Parses ground truth plans from ToolHop format"""
    
    @staticmethod
    def parse_toolhop_example(example: Dict) -> Plan:
        """
        Parse a ToolHop example to extract the ground truth plan.
        
        The ground truth plan is derived from the sub_task structure.
        """
        steps = []
        sub_tasks = example["sub_task"]
        
        # Create mapping of sub-task question to tool
        for step_id, (question, answer) in enumerate(sub_tasks.items()):
            if question in example["tools"]:
                tool_spec = example["tools"][question]
                tool_name = tool_spec["name"]
                
                # Extract parameters - this is simplified, would need more logic
                # For now, we'll create a basic parameter structure
                parameters = GroundTruthParser._extract_parameters(
                    question, answer, tool_spec, step_id
                )
                
                step = ToolCall(
                    step_id=step_id,
                    tool_name=tool_name,
                    parameters=parameters,
                    output_variable=f"{{{{{step_id}}}}}",
                    expected_output=str(answer)
                )
                steps.append(step)
        
        return Plan(steps=steps, error_type=ErrorType.NONE)
    
    @staticmethod
    def _extract_parameters(question: str, answer: str, tool_spec: Dict, step_id: int) -> Dict[str, Any]:
        """Extract parameters for a tool call based on the question and answer"""
        parameters = {}
        
        # Extract key entities from the question
        # This is a simplified version - you'd want more sophisticated NLP here
        
        # For the first step, parameters come from the question
        if step_id == 0:
            # Extract the main entity from the question
            # e.g., "Salisbury Woodland Gardens links a zoo with which park?"
            # -> location_name = "Salisbury Woodland Gardens"
            words = question.split()
            if tool_spec["name"] == "geo_relationship_finder":
                # Find the location name (usually the first proper noun sequence)
                location_parts = []
                for word in words[:5]:  # Check first few words
                    if word[0].isupper() and word not in ["What", "Which", "Who", "How"]:
                        location_parts.append(word)
                    elif location_parts:
                        break
                if location_parts:
                    parameters["location_name"] = " ".join(location_parts)
                # Add entity types based on question
                if "zoo" in question.lower():
                    parameters["entity_types"] = ["zoo", "park"]
        else:
            # For subsequent steps, parameters reference previous outputs
            # This requires dependency tracking
            parameters = GroundTruthParser._infer_parameters(question, answer, tool_spec, step_id)
        
        return parameters
    
    @staticmethod
    def _infer_parameters(question: str, answer: str, tool_spec: Dict, step_id: int) -> Dict[str, Any]:
        """Infer parameters for non-first steps"""
        parameters = {}
        required_params = tool_spec["parameters"].get("required", [])
        
        # Check if we need to reference a previous step's output
        if step_id > 0:
            # Add dependency reference
            for param in required_params:
                param_spec = tool_spec["parameters"]["properties"].get(param, {})
                if "description" in param_spec:
                    # Use the previous step's output
                    if tool_spec["name"] == "historical_figure_identifier":
                        parameters["event_name"] = f"{{{{{step_id - 1}}}}}"
                    elif tool_spec["name"] == "extract_first_name":
                        parameters["full_name"] = f"{{{{{step_id - 1}}}}}"
                    elif tool_spec["name"] == "count_letters":
                        parameters["input"] = f"{{{{{step_id - 1}}}}}"
                        # Parse special requirements from the question
                        if "exclude the first and last" in question.lower():
                            parameters["ignore_position"] = ["first", "last"]
        
        return parameters


class StaticValidator:
    """Static validation of plans without execution"""
    
    def __init__(self, available_tools: Dict[str, Dict]):
        self.available_tools = available_tools
    
    def validate(self, plan: Plan, query: str) -> Tuple[bool, List[str]]:
        """
        Validate a plan statically.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check 1: Tool existence
        for step in plan.steps:
            if not self._tool_exists(step.tool_name):
                errors.append(f"Step {step.step_id}: Tool '{step.tool_name}' does not exist")
        
        # Check 2: Required parameters
        for step in plan.steps:
            missing_params = self._check_required_parameters(step)
            if missing_params:
                errors.append(f"Step {step.step_id}: Missing required parameters: {missing_params}")
        
        # Check 3: Type compatibility (basic check)
        for step in plan.steps:
            type_errors = self._check_types(step)
            errors.extend([f"Step {step.step_id}: {err}" for err in type_errors])
        
        # Check 4: Dependency validation
        dep_errors = self._check_dependencies(plan)
        errors.extend(dep_errors)
        
        # Check 5: Circular dependencies
        circular = self._check_circular_dependencies(plan)
        if circular:
            errors.append(f"Circular dependency detected: {circular}")
        
        return len(errors) == 0, errors
    
    def _tool_exists(self, tool_name: str) -> bool:
        """Check if tool exists in available tools"""
        return any(tool_spec["name"] == tool_name for tool_spec in self.available_tools.values())
    
    def _check_required_parameters(self, step: ToolCall) -> List[str]:
        """Check if all required parameters are present"""
        tool_spec = self._get_tool_spec(step.tool_name)
        if not tool_spec:
            return []
        
        required = tool_spec["parameters"].get("required", [])
        provided = set(step.parameters.keys())
        missing = [param for param in required if param not in provided]
        return missing
    
    def _check_types(self, step: ToolCall) -> List[str]:
        """Basic type checking for parameters"""
        errors = []
        tool_spec = self._get_tool_spec(step.tool_name)
        if not tool_spec:
            return errors
        
        properties = tool_spec["parameters"].get("properties", {})
        for param_name, param_value in step.parameters.items():
            if param_name in properties:
                expected_type = properties[param_name].get("type")
                
                # Skip validation for dependency references like "{{0}}"
                if isinstance(param_value, str) and re.match(r"\{\{\d+\}\}", param_value):
                    continue
                
                # Basic type checking
                if expected_type == "string" and not isinstance(param_value, str):
                    errors.append(f"Parameter '{param_name}' should be string, got {type(param_value).__name__}")
                elif expected_type == "number" and not isinstance(param_value, (int, float)):
                    errors.append(f"Parameter '{param_name}' should be number, got {type(param_value).__name__}")
                elif expected_type == "boolean" and not isinstance(param_value, bool):
                    errors.append(f"Parameter '{param_name}' should be boolean, got {type(param_value).__name__}")
                elif expected_type == "array" and not isinstance(param_value, list):
                    errors.append(f"Parameter '{param_name}' should be array, got {type(param_value).__name__}")
        
        return errors
    
    def _check_dependencies(self, plan: Plan) -> List[str]:
        """Check that dependency references are valid"""
        errors = []
        max_step_id = max(step.step_id for step in plan.steps) if plan.steps else -1
        
        for step in plan.steps:
            for param_value in step.parameters.values():
                if isinstance(param_value, str):
                    # Find all {{N}} references
                    refs = re.findall(r"\{\{(\d+)\}\}", param_value)
                    for ref in refs:
                        ref_id = int(ref)
                        if ref_id >= step.step_id:
                            errors.append(f"Step {step.step_id}: Invalid forward reference to step {ref_id}")
                        if ref_id > max_step_id:
                            errors.append(f"Step {step.step_id}: Reference to non-existent step {ref_id}")
        
        return errors
    
    def _check_circular_dependencies(self, plan: Plan) -> Optional[str]:
        """Check for circular dependencies"""
        # Build dependency graph
        graph = defaultdict(list)
        for step in plan.steps:
            deps = []
            for param_value in step.parameters.values():
                if isinstance(param_value, str):
                    refs = re.findall(r"\{\{(\d+)\}\}", param_value)
                    deps.extend([int(ref) for ref in refs])
            graph[step.step_id] = deps
        
        # DFS to detect cycles
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        visited = set()
        for step in plan.steps:
            if step.step_id not in visited:
                if has_cycle(step.step_id, visited, set()):
                    return f"Cycle involving step {step.step_id}"
        
        return None
    
    def _get_tool_spec(self, tool_name: str) -> Optional[Dict]:
        """Get tool specification by name"""
        for tool_spec in self.available_tools.values():
            if tool_spec["name"] == tool_name:
                return tool_spec
        return None


class PlanGenerator:
    """Generates candidate plans using LLM"""
    
    def __init__(self, api_key: str, model: str = "gpt-4.1-mini", use_anthropic: bool = False):
        self.api_key = api_key
        self.model = model
        self.use_anthropic = use_anthropic
        
        if use_anthropic:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = openai.OpenAI(api_key=api_key)
    
    def generate_candidates(
        self,
        query: str,
        tools: Dict[str, Dict],
        ground_truth: Plan,
        n_candidates: int = 10,
        error_distribution: Optional[Dict[ErrorType, float]] = None
    ) -> List[Plan]:
        """
        Generate N candidate plans with diverse error types.
        
        Args:
            query: The user query
            tools: Available tools
            ground_truth: The ground truth plan
            n_candidates: Number of candidates to generate
            error_distribution: Distribution of error types (if None, uses default)
        
        Returns:
            List of candidate plans
        """
        if error_distribution is None:
            error_distribution = {
                ErrorType.NONE: 0.1,  # 10% high quality
                ErrorType.TYPE_MISMATCH: 0.1,
                ErrorType.MISSING_DEPENDENCY: 0.1,
                ErrorType.WRONG_TOOL: 0.15,
                ErrorType.PARAMETER_TYPO: 0.15,
                ErrorType.INEFFICIENT_ORDER: 0.15,
                ErrorType.INCOMPLETE_PLAN: 0.1,
                ErrorType.UNNECESSARY_STEPS: 0.1,
                ErrorType.CIRCULAR_DEPENDENCY: 0.05
            }
        
        candidates = []
        
        # Always include ground truth
        candidates.append(ground_truth)
        
        # Generate candidates for each error type based on distribution
        error_counts = {
            error_type: max(1, int(n_candidates * prob))
            for error_type, prob in error_distribution.items()
            if error_type != ErrorType.NONE
        }
        
        for error_type, count in error_counts.items():
            for _ in range(count):
                if len(candidates) >= n_candidates:
                    break
                
                candidate = self._generate_plan_with_error(
                    query, tools, ground_truth, error_type
                )
                candidates.append(candidate)
        
        # Fill remaining with random error types
        while len(candidates) < n_candidates:
            error_type = random.choice([e for e in ErrorType if e != ErrorType.NONE])
            candidate = self._generate_plan_with_error(
                query, tools, ground_truth, error_type
            )
            candidates.append(candidate)
        
        return candidates[:n_candidates]
    
    def _generate_plan_with_error(
        self,
        query: str,
        tools: Dict[str, Dict],
        ground_truth: Plan,
        error_type: ErrorType
    ) -> Plan:
        """Generate a single plan with a specific error type"""
        
        if error_type == ErrorType.TYPE_MISMATCH:
            return self._inject_type_mismatch(ground_truth, tools)
        elif error_type == ErrorType.MISSING_DEPENDENCY:
            return self._inject_missing_dependency(ground_truth)
        elif error_type == ErrorType.WRONG_TOOL:
            return self._inject_wrong_tool(ground_truth, tools, query)
        elif error_type == ErrorType.PARAMETER_TYPO:
            return self._inject_parameter_typo(ground_truth)
        elif error_type == ErrorType.INEFFICIENT_ORDER:
            return self._inject_inefficient_order(ground_truth)
        elif error_type == ErrorType.INCOMPLETE_PLAN:
            return self._inject_incomplete_plan(ground_truth)
        elif error_type == ErrorType.UNNECESSARY_STEPS:
            return self._inject_unnecessary_steps(ground_truth, tools)
        elif error_type == ErrorType.CIRCULAR_DEPENDENCY:
            return self._inject_circular_dependency(ground_truth)
        else:
            return ground_truth
    
    def _inject_type_mismatch(self, plan: Plan, tools: Dict) -> Plan:
        """Inject a type mismatch error"""
        new_steps = []
        for i, step in enumerate(plan.steps):
            new_step = ToolCall(
                step_id=step.step_id,
                tool_name=step.tool_name,
                parameters=step.parameters.copy(),
                output_variable=step.output_variable,
                expected_output=step.expected_output
            )
            
            # Change a parameter type for one step
            if i == len(plan.steps) // 2 and new_step.parameters:
                param_name = list(new_step.parameters.keys())[0]
                # Convert string to number or vice versa
                if isinstance(new_step.parameters[param_name], str):
                    new_step.parameters[param_name] = 42
                else:
                    new_step.parameters[param_name] = "invalid"
            
            new_steps.append(new_step)
        
        return Plan(steps=new_steps, error_type=ErrorType.TYPE_MISMATCH)
    
    def _inject_missing_dependency(self, plan: Plan) -> Plan:
        """Remove a dependency reference"""
        new_steps = []
        for i, step in enumerate(plan.steps):
            new_step = ToolCall(
                step_id=step.step_id,
                tool_name=step.tool_name,
                parameters=step.parameters.copy(),
                output_variable=step.output_variable,
                expected_output=step.expected_output
            )
            
            # Remove dependency reference for one step
            if i > 0 and i == len(plan.steps) // 2:
                for param_name, param_value in new_step.parameters.items():
                    if isinstance(param_value, str) and "{{" in param_value:
                        # Replace dependency with a literal value
                        new_step.parameters[param_name] = "literal_value"
                        break
            
            new_steps.append(new_step)
        
        return Plan(steps=new_steps, error_type=ErrorType.MISSING_DEPENDENCY)
    
    def _inject_wrong_tool(self, plan: Plan, tools: Dict, query: str) -> Plan:
        """Use the wrong tool for a step"""
        new_steps = []
        tool_names = [spec["name"] for spec in tools.values()]
        
        for i, step in enumerate(plan.steps):
            if i == len(plan.steps) // 2:
                # Pick a different tool
                other_tools = [t for t in tool_names if t != step.tool_name]
                if other_tools:
                    wrong_tool = random.choice(other_tools)
                    new_step = ToolCall(
                        step_id=step.step_id,
                        tool_name=wrong_tool,
                        parameters=step.parameters.copy(),
                        output_variable=step.output_variable,
                        expected_output=step.expected_output
                    )
                    new_steps.append(new_step)
                    continue
            
            new_steps.append(step)
        
        return Plan(steps=new_steps, error_type=ErrorType.WRONG_TOOL)
    
    def _inject_parameter_typo(self, plan: Plan) -> Plan:
        """Inject a typo in a parameter value"""
        new_steps = []
        for i, step in enumerate(plan.steps):
            new_step = ToolCall(
                step_id=step.step_id,
                tool_name=step.tool_name,
                parameters=step.parameters.copy(),
                output_variable=step.output_variable,
                expected_output=step.expected_output
            )
            
            # Add typo to one parameter
            if i == 0 and new_step.parameters:
                for param_name, param_value in new_step.parameters.items():
                    if isinstance(param_value, str) and len(param_value) > 3 and "{{" not in param_value:
                        # Insert/delete/swap a character
                        typo_type = random.choice(["insert", "delete", "swap"])
                        if typo_type == "insert":
                            pos = random.randint(0, len(param_value))
                            new_step.parameters[param_name] = param_value[:pos] + "x" + param_value[pos:]
                        elif typo_type == "delete" and len(param_value) > 1:
                            pos = random.randint(0, len(param_value) - 1)
                            new_step.parameters[param_name] = param_value[:pos] + param_value[pos+1:]
                        elif typo_type == "swap" and len(param_value) > 1:
                            pos = random.randint(0, len(param_value) - 2)
                            chars = list(param_value)
                            chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
                            new_step.parameters[param_name] = "".join(chars)
                        break
            
            new_steps.append(new_step)
        
        return Plan(steps=new_steps, error_type=ErrorType.PARAMETER_TYPO)
    
    def _inject_inefficient_order(self, plan: Plan) -> Plan:
        """Reorder steps inefficiently"""
        if len(plan.steps) < 2:
            return Plan(steps=plan.steps.copy(), error_type=ErrorType.INEFFICIENT_ORDER)
        
        # Swap two adjacent steps if possible (without breaking dependencies)
        new_steps = plan.steps.copy()
        # Simple shuffle that maintains some dependencies
        if len(new_steps) >= 3:
            # Swap last two steps
            new_steps[-1], new_steps[-2] = new_steps[-2], new_steps[-1]
            # Update step IDs
            for i, step in enumerate(new_steps):
                step.step_id = i
        
        return Plan(steps=new_steps, error_type=ErrorType.INEFFICIENT_ORDER)
    
    def _inject_incomplete_plan(self, plan: Plan) -> Plan:
        """Remove some steps from the plan"""
        if len(plan.steps) <= 1:
            return Plan(steps=plan.steps.copy(), error_type=ErrorType.INCOMPLETE_PLAN)
        
        # Remove the last step
        new_steps = plan.steps[:-1]
        return Plan(steps=new_steps, error_type=ErrorType.INCOMPLETE_PLAN)
    
    def _inject_unnecessary_steps(self, plan: Plan, tools: Dict) -> Plan:
        """Add unnecessary steps"""
        new_steps = plan.steps.copy()
        
        # Add a redundant step in the middle
        if len(plan.steps) > 1:
            insert_pos = len(plan.steps) // 2
            # Duplicate an existing step
            redundant_step = ToolCall(
                step_id=insert_pos,
                tool_name=plan.steps[0].tool_name,
                parameters=plan.steps[0].parameters.copy(),
                output_variable=f"{{{{{insert_pos}}}}}",
                expected_output=None
            )
            new_steps.insert(insert_pos, redundant_step)
            
            # Update step IDs
            for i, step in enumerate(new_steps):
                step.step_id = i
        
        return Plan(steps=new_steps, error_type=ErrorType.UNNECESSARY_STEPS)
    
    def _inject_circular_dependency(self, plan: Plan) -> Plan:
        """Create a circular dependency"""
        if len(plan.steps) < 2:
            return Plan(steps=plan.steps.copy(), error_type=ErrorType.CIRCULAR_DEPENDENCY)
        
        new_steps = []
        for i, step in enumerate(plan.steps):
            new_step = ToolCall(
                step_id=step.step_id,
                tool_name=step.tool_name,
                parameters=step.parameters.copy(),
                output_variable=step.output_variable,
                expected_output=step.expected_output
            )
            
            # Make step i depend on step i+1 (creating a cycle)
            if i == 0 and len(plan.steps) > 1 and new_step.parameters:
                # Add forward reference
                param_name = list(new_step.parameters.keys())[0]
                new_step.parameters[param_name] = "{{1}}"
            
            new_steps.append(new_step)
        
        return Plan(steps=new_steps, error_type=ErrorType.CIRCULAR_DEPENDENCY)


class LLMJudgeAnnotator:
    """Uses LLM to annotate plans with quality scores and reasoning"""
    
    def __init__(self, api_key: str, model: str = "gpt-4.1-mini"):
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
    
    def annotate_plan(
        self,
        query: str,
        tools: Dict[str, Dict],
        plan: Plan,
        ground_truth: Optional[Plan] = None
    ) -> JudgeAnnotation:
        """
        Annotate a plan with quality score, success prediction, and reasoning.
        
        Args:
            query: The user query
            tools: Available tools
            plan: The plan to annotate
            ground_truth: Optional ground truth plan for comparison
        
        Returns:
            JudgeAnnotation with scores and analysis
        """
        prompt = self._build_annotation_prompt(query, tools, plan, ground_truth)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating multi-step tool execution plans. Analyze plans carefully and provide detailed, structured feedback."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_completion_tokens=2000
            )
            
            content = response.choices[0].message.content
            annotation = self._parse_annotation(content)
            return annotation
            
        except Exception as e:
            print(f"Error calling LLM: {e}")
            # Return default annotation on error
            return JudgeAnnotation(
                quality_score=50,
                success_prediction="uncertain",
                reasoning="Error during annotation",
                issues=[],
                confidence=0.0
            )
    
    def _build_annotation_prompt(
        self,
        query: str,
        tools: Dict[str, Dict],
        plan: Plan,
        ground_truth: Optional[Plan]
    ) -> str:
        """Build the prompt for LLM annotation"""
        
        tools_desc = self._format_tools(tools)
        plan_desc = self._format_plan(plan)
        
        prompt = f"""You are evaluating a multi-step tool execution plan using an OBJECTIVE ERROR-BASED RUBRIC.

**USER QUERY:**
{query}

**AVAILABLE TOOLS:**
{tools_desc}

**PROPOSED PLAN:**
{plan_desc}

{"**GROUND TRUTH PLAN (for reference):**" + chr(10) + self._format_plan(ground_truth) if ground_truth else ""}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OBJECTIVE SCORING RUBRIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**SCORING FORMULA:**
1. Start with a base score of 100
2. Identify ALL errors in the plan
3. For EACH error, deduct points based on severity:
   - CRITICAL error: -30 points
   - HIGH error: -20 points
   - MEDIUM error: -10 points
   - LOW error: -5 points
4. Final score = max(0, 100 - total_deductions)

**CRITICAL: Understanding Plan Dependencies and Parameters**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO CHECK IF PARAMETERS ARE PROVIDED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In multi-step plans, parameters can be provided in TWO ways:

1. **Literal values:** `location_name="London"` or `location_name='London'`
2. **Dependency references:** `event_name='{{{{0}}}}'` or `event_name="{{{{0}}}}"`

BOTH formats mean the parameter IS PROVIDED. The notation {{{{N}}}} refers to 
the output from step N.

**CORRECT EXAMPLES:**

✅ Step 0: {{{{0}}}} = search(query="London")
   → Parameter "query" IS PROVIDED with literal value "London"

✅ Step 1: {{{{1}}}} = summarize(text='{{{{0}}}}')  
   → Parameter "text" IS PROVIDED with reference to step 0's output

✅ Step 2: {{{{2}}}} = translate(text="{{{{1}}}}", target_lang="French")
   → Both parameters ARE PROVIDED (one reference, one literal)

**INCORRECT EXAMPLES:**

❌ Step 1: {{{{1}}}} = summarize()
   → Parameter "text" is MISSING (no parameter provided at all)

❌ Step 2: {{{{2}}}} = translate(target_lang="French")
   → Parameter "text" is MISSING (only one param when two are required)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: SYNTACTIC vs SEMANTIC VALIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your job is to check SYNTACTIC correctness, NOT SEMANTIC correctness.

**SYNTACTIC (what you SHOULD check):**
✅ Are all required parameters provided?
✅ Are parameter types correct (string vs int vs array)?
✅ Are dependencies valid (no forward refs, no circular deps)?
✅ Are all steps present to complete the task?
✅ Is the tool name spelled correctly?

**SEMANTIC (what you should NOT check):**
❌ Will the tool outputs actually work together?
❌ Is the output of tool A meaningful for tool B?
❌ Will the plan produce the correct final answer?
❌ Is the data flow logically sound?

**Example:**

Step 0: {{0}} = geo_relationship_finder(location_name='Gardens')
Step 1: {{1}} = historical_figure_identifier(event_name='{{0}}')

❌ WRONG: "{{0}} is a list of locations, not an event name - semantic error!"
✅ CORRECT: "All required parameters provided, dependency reference valid - no error"

You are NOT evaluating whether the plan will work correctly. You are ONLY 
evaluating whether the plan is syntactically valid (proper structure, types, 
dependencies). Assume the person writing the plan knows what they're doing 
semantically.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO CHECK FOR MISSING PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Step-by-step process:**

1. Look at the tool's "Required parameters" list
2. For each required parameter, check if it appears in the step
3. A parameter is PROVIDED if you see: `param_name=<any_value>`
   - `<any_value>` can be: "literal", 'literal', {{{{N}}}}, '{{{{N}}}}', ["array"], etc.
4. A parameter is MISSING only if the parameter name doesn't appear at all

**Example:**

Tool: historical_figure_identifier
Required parameters: event_name

Step 1: {{{{1}}}} = historical_figure_identifier(event_name='{{{{0}}}}')
                                              ^^^^^^^^^^^^^^^^
                                              Parameter IS PROVIDED!

Check: ✅ event_name appears with value '{{{{0}}}}'
Result: NO MISSING PARAMETER ERROR

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHEN TO FLAG {{{{N}}}} REFERENCES AS ERRORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{{{{N}}}} references are CORRECT unless:

❌ Forward reference: Step 1 uses {{{{2}}}} (references future step)
❌ Circular reference: Step 0 uses {{{{0}}}} (references itself)  
❌ Non-existent step: Step 2 uses {{{{5}}}} (only 3 steps exist)
❌ Should use {{{{N}}}} but uses literal: Step 2 uses "some_text" instead of {{{{1}}}}

✅ CORRECT: Step 1 uses {{{{0}}}} (references previous step)
✅ CORRECT: Step 2 uses {{{{1}}}} (references previous step)
✅ CORRECT: Step 3 uses {{{{0}}}} and {{{{2}}}} (references multiple previous steps)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULE: ONLY CHECK REQUIRED PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DO NOT flag missing parameters unless they are in the "Required parameters" list.

Example:

Tool: geo_relationship_finder  
Required parameters: location_name, entity_types

Step 0: {{{{0}}}} = geo_relationship_finder(location_name='Gardens', entity_types=['zoo'])

✅ CORRECT: Both required parameters are provided
❌ WRONG: Don't complain about missing "radius" - it's not required!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: Avoid Double-Counting Errors
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If ONE mistake has multiple symptoms, classify by the MOST SEVERE impact 
and deduct points ONLY ONCE.

**WRONG Approach (Double-Counting):**
  Step 2: full_name=42 instead of full_name='{{{{1}}}}'
  
  Issue 1: [HIGH] Missing dependency - uses literal instead of {{{{1}}}} (-20 pts)
  Issue 2: [CRITICAL] Type mismatch - uses int instead of string (-30 pts)
  Total deduction: -50 points ❌ INCORRECT!

**CORRECT Approach (Single Deduction):**
  Step 2: full_name=42 instead of full_name='{{{{1}}}}'
  
  Issue 1: [CRITICAL] Type mismatch AND missing dependency (-30 pts)
  Reasoning: "This error breaks the dependency chain AND causes a type 
              mismatch, but it's ONE mistake with multiple effects."
  Total deduction: -30 points ✅ CORRECT!

**Rule:** When analyzing an error, ask yourself: "Is this ONE mistake with 
multiple consequences, or MULTIPLE separate mistakes?" If it's one mistake, 
list it as one issue with the highest severity.

**More Examples:**

✅ CORRECT: Step using wrong tool with wrong parameters
  → ONE issue: [CRITICAL] Wrong tool selected (-30 pts)
  → The parameters being wrong is a CONSEQUENCE of wrong tool

✅ CORRECT: Step missing two different required parameters  
  → TWO issues: [CRITICAL] Missing param1 (-30), [CRITICAL] Missing param2 (-30)
  → These are separate mistakes

✅ CORRECT: Step has typo AND wrong tool
  → TWO issues: [MEDIUM] Typo (-10), [CRITICAL] Wrong tool (-30)
  → These are separate mistakes

**ERROR SEVERITY DEFINITIONS:**

┌─────────────────────────────────────────────────────────────────┐
│ CRITICAL SEVERITY (-30 points each)                             │
├─────────────────────────────────────────────────────────────────┤
│ • Missing required parameters (tool cannot execute)             │
│ • Circular dependency ({{{{N}}}} references itself/cycle)           │
│ • Non-existent tool used                                        │
│ • Forward reference ({{{{N}}}} where N >= current step)             │
│ • Wrong tool that cannot produce needed output                  │
│ • Type mismatch that causes execution failure                   │
│ • Missing critical steps (plan cannot achieve goal)             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ HIGH SEVERITY (-20 points each)                                 │
├─────────────────────────────────────────────────────────────────┤
│ • Missing dependency: hardcoded value instead of {{{{N}}}}          │
│ • Wrong tool but parameters work (semantic incorrectness)       │
│ • Type mismatch that may work but is incorrect                  │
│ • Incorrect parameter value that changes output significantly   │
│ • Missing non-critical but important step                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ MEDIUM SEVERITY (-10 points each)                               │
├─────────────────────────────────────────────────────────────────┤
│ • Inefficient step ordering (works but suboptimal)              │
│ • Typo in parameter value (might still work)                    │
│ • Unnecessary step that doesn't break the plan                  │
│ • Redundant computation                                         │
│ • Suboptimal tool choice (correct output, but inefficient)      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ LOW SEVERITY (-5 points each)                                   │
├─────────────────────────────────────────────────────────────────┤
│ • Minor formatting issues                                       │
│ • Non-standard but functional parameter format                  │
│ • Minor inefficiency that barely impacts performance            │
│ • Style issues (but plan is correct)                            │
└─────────────────────────────────────────────────────────────────┘

**ERROR TYPE TO SEVERITY MAPPING GUIDE:**

| Error Type              | Typical Severity | Examples                          |
|------------------------|------------------|-----------------------------------|
| type_mismatch          | CRITICAL/HIGH    | string → int causes crash         |
| missing_dependency     | HIGH             | "London" instead of {{{{1}}}}         |
| wrong_tool             | CRITICAL/HIGH    | calculator instead of search      |
| parameter_typo         | MEDIUM/LOW       | "Londno" vs "London"              |
| circular_dependency    | CRITICAL         | {{{{0}}}} references {{{{0}}}}            |
| inefficient_order      | MEDIUM           | Step 3 before Step 2              |
| incomplete_plan        | CRITICAL         | Missing final step                |
| unnecessary_steps      | MEDIUM           | Redundant duplicate step          |
| forward_reference      | CRITICAL         | Step 2 uses {{{{3}}}}                 |

**SUCCESS PREDICTION MAPPING:**
Based on final quality score:
- 90-100: "yes" (will definitely succeed)
- 75-89: "likely_yes" (probably will succeed)
- 50-74: "uncertain" (could go either way)
- 25-49: "likely_no" (probably will fail)
- 0-24: "no" (will definitely fail)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **IDENTIFY ALL ERRORS**: Go through the plan step-by-step
2. **CLASSIFY SEVERITY**: Use the rubric above
3. **CALCULATE SCORE**: Start at 100, deduct points per error
4. **SHOW YOUR WORK**: In reasoning, list each error and deduction

**RESPONSE FORMAT (JSON only, no markdown):**

{{
  "quality_score": <0-100>,
  "success_prediction": "<yes|likely_yes|uncertain|likely_no|no>",
  "confidence": <0.0-1.0>,
  "reasoning": "Base score: 100\\nError 1: [severity] description (-X points)\\nError 2: [severity] description (-X points)\\n...\\nFinal score: 100 - X - Y = Z",
  "issues": [
    {{
      "type": "type_mismatch|missing_dependency|wrong_tool|parameter_typo|circular_dependency|inefficient_order|incomplete_plan|unnecessary_steps|forward_reference|other",
      "severity": "critical|high|medium|low",
      "step": <step_number or null>,
      "description": "Specific description of the error",
      "suggestion": "How to fix this error",
      "points_deducted": <30|20|10|5>
    }}
  ]
}}

**IMPORTANT SCORING RULES:**
1. If plan has 0 errors → Score = 95-100 (perfect plan)
2. Be consistent: Same error type = same severity = same deduction
3. Count EACH occurrence separately (3 type errors = 3 deductions)
4. If multiple errors in same step, count them separately ONLY if they are truly separate mistakes
5. Show your calculation clearly in reasoning

Begin your analysis:"""
        
        return prompt
    
    def _format_tools(self, tools: Dict[str, Dict]) -> str:
        """Format tools for the prompt"""
        lines = []
        for tool_spec in tools.values():
            name = tool_spec["name"]
            desc = tool_spec["description"]
            params = tool_spec["parameters"]
            required = params.get("required", [])
            
            lines.append(f"\n**{name}**")
            lines.append(f"Description: {desc}")
            lines.append(f"Required parameters: {', '.join(required)}")
            
            # Show parameter details
            for param_name, param_spec in params.get("properties", {}).items():
                param_type = param_spec.get("type", "unknown")
                param_desc = param_spec.get("description", "")
                lines.append(f"  - {param_name} ({param_type}): {param_desc}")
        
        return "\n".join(lines)
    
    def _format_plan(self, plan: Plan) -> str:
        """Format plan for the prompt"""
        if not plan:
            return "N/A"
        
        lines = []
        for step in plan.steps:
            lines.append(str(step))
        
        if plan.error_type != ErrorType.NONE:
            lines.append(f"\n[Note: This plan was generated with error type: {plan.error_type.value}]")
        
        return "\n".join(lines)
    
    def _parse_annotation(self, content: str) -> JudgeAnnotation:
        """Parse LLM response into JudgeAnnotation"""
        try:
            # Extract JSON from markdown code block if present
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")
            
            data = json.loads(json_str)
            
            # Validate quality score is in range
            quality_score = data.get("quality_score", 50)
            if quality_score < 0 or quality_score > 100:
                print(f"Warning: Quality score {quality_score} out of range, clipping to [0, 100]")
                quality_score = max(0, min(100, quality_score))
            
            return JudgeAnnotation(
                quality_score=quality_score,
                success_prediction=data.get("success_prediction", "uncertain"),
                reasoning=data.get("reasoning", ""),
                issues=data.get("issues", []),
                confidence=data.get("confidence", 0.5)
            )
            
        except Exception as e:
            print(f"Error parsing annotation: {e}")
            print(f"Content: {content[:500]}...")  # Show first 500 chars
            # Return default
            return JudgeAnnotation(
                quality_score=50,
                success_prediction="uncertain",
                reasoning=f"Parse error: {str(e)}\nRaw content: {content[:200]}",
                issues=[],
                confidence=0.5
            )


class DatasetGenerator:
    """Main class to generate the complete dataset"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-mini",
        use_anthropic: bool = False
    ):
        self.plan_generator = PlanGenerator(api_key, model, use_anthropic)
        self.annotator = LLMJudgeAnnotator(api_key, model)
    
    def generate_dataset(
        self,
        toolhop_path: str,
        output_path: str,
        n_candidates_per_query: int = 10,
        max_queries: Optional[int] = None,
        validate: bool = True
    ):
        """
        Generate the complete dataset from ToolHop.
        
        Args:
            toolhop_path: Path to ToolHop JSON file
            output_path: Path to save the generated dataset
            n_candidates_per_query: Number of candidate plans per query
            max_queries: Maximum number of queries to process (None = all)
            validate: Whether to run validation checks
        """
        print("Loading ToolHop dataset...")
        with open(toolhop_path, 'r') as f:
            toolhop_data = json.load(f)
        
        if max_queries:
            toolhop_data = toolhop_data[:max_queries]
        
        print(f"Processing {len(toolhop_data)} queries...")
        
        all_annotated_plans = []
        
        for example in tqdm(toolhop_data, desc="Generating plans"):
            query_id = example["id"]
            query = example["question"]
            tools = example["tools"]
            
            # Parse ground truth
            print(f"\n{'='*80}")
            print(f"Query {query_id}: {query}")
            print(f"{'='*80}")
            
            ground_truth = GroundTruthParser.parse_toolhop_example(example)
            print(f"\nGround truth plan ({len(ground_truth.steps)} steps):")
            print(ground_truth)
            
            # Generate candidates
            print(f"\nGenerating {n_candidates_per_query} candidate plans...")
            candidates = self.plan_generator.generate_candidates(
                query, tools, ground_truth, n_candidates_per_query
            )
            
            # Static validation
            print("\nRunning static validation...")
            validator = StaticValidator(tools)
            for i, candidate in enumerate(candidates):
                is_valid, errors = validator.validate(candidate, query)
                status = "✓ VALID" if is_valid else f"✗ INVALID ({len(errors)} errors)"
                print(f"  Plan {i} [{candidate.error_type.value}]: {status}")
                if errors and validate:
                    for error in errors[:3]:  # Show first 3 errors
                        print(f"    - {error}")
            
            # Annotate with LLM judge
            print(f"\nAnnotating plans with LLM judge...")
            for i, candidate in enumerate(tqdm(candidates, desc="Annotating", leave=False)):
                annotation = self.annotator.annotate_plan(
                    query, tools, candidate, ground_truth
                )
                
                annotated_plan = AnnotatedPlan(
                    plan=candidate,
                    annotation=annotation,
                    query_id=query_id
                )
                all_annotated_plans.append(annotated_plan)
                
                if validate and i < 3:  # Show details for first 3 plans
                    print(f"\n  Plan {i} [{candidate.error_type.value}]:")
                    print(f"    Quality: {annotation.quality_score}/100")
                    print(f"    Success: {annotation.success_prediction}")
                    print(f"    Issues: {len(annotation.issues)}")
                    if annotation.issues:
                        for issue in annotation.issues[:2]:
                            print(f"      - [{issue['severity']}] {issue['description']}")
        
        # Save dataset
        print(f"\n{'='*80}")
        print(f"Saving dataset to {output_path}...")
        dataset = {
            "metadata": {
                "n_queries": len(toolhop_data),
                "n_candidates_per_query": n_candidates_per_query,
                "total_plans": len(all_annotated_plans),
                "model": self.annotator.model
            },
            "data": [plan.to_dict() for plan in all_annotated_plans]
        }
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✓ Dataset saved: {len(all_annotated_plans)} annotated plans")
        
        # Validation report
        if validate:
            self._print_validation_report(all_annotated_plans)
    
    def _print_validation_report(self, annotated_plans: List[AnnotatedPlan]):
        """Print a validation report of the generated dataset"""
        print(f"\n{'='*80}")
        print("VALIDATION REPORT")
        print(f"{'='*80}")
        
        # Group by error type
        error_type_counts = defaultdict(int)
        error_type_scores = defaultdict(list)
        error_type_issues = defaultdict(list)
        
        for ap in annotated_plans:
            error_type = ap.plan.error_type
            error_type_counts[error_type] += 1
            error_type_scores[error_type].append(ap.annotation.quality_score)
            error_type_issues[error_type].append(len(ap.annotation.issues))
        
        print("\nError Type Distribution:")
        for error_type in ErrorType:
            count = error_type_counts[error_type]
            if count > 0:
                avg_score = sum(error_type_scores[error_type]) / len(error_type_scores[error_type])
                avg_issues = sum(error_type_issues[error_type]) / len(error_type_issues[error_type])
                print(f"  {error_type.value:25s}: {count:4d} plans (avg score: {avg_score:5.1f}, avg issues: {avg_issues:.1f})")
        
        # Quality score distribution
        all_scores = [ap.annotation.quality_score for ap in annotated_plans]
        print(f"\nQuality Score Statistics:")
        print(f"  Mean:   {sum(all_scores)/len(all_scores):.1f}")
        print(f"  Median: {sorted(all_scores)[len(all_scores)//2]:.1f}")
        print(f"  Min:    {min(all_scores):.1f}")
        print(f"  Max:    {max(all_scores):.1f}")
        
        # Score distribution by ranges
        score_ranges = {
            "90-100 (Excellent)": 0,
            "75-89 (Good)": 0,
            "50-74 (Fair)": 0,
            "25-49 (Poor)": 0,
            "0-24 (Critical)": 0
        }
        for score in all_scores:
            if score >= 90:
                score_ranges["90-100 (Excellent)"] += 1
            elif score >= 75:
                score_ranges["75-89 (Good)"] += 1
            elif score >= 50:
                score_ranges["50-74 (Fair)"] += 1
            elif score >= 25:
                score_ranges["25-49 (Poor)"] += 1
            else:
                score_ranges["0-24 (Critical)"] += 1
        
        print(f"\nScore Distribution:")
        for range_label, count in score_ranges.items():
            pct = 100 * count / len(all_scores)
            print(f"  {range_label:20s}: {count:4d} ({pct:5.1f}%)")
        
        # Success prediction distribution
        success_counts = defaultdict(int)
        for ap in annotated_plans:
            success_counts[ap.annotation.success_prediction] += 1
        
        print(f"\nSuccess Prediction Distribution:")
        for pred in ["yes", "likely_yes", "uncertain", "likely_no", "no"]:
            count = success_counts[pred]
            pct = 100 * count / len(annotated_plans)
            print(f"  {pred:12s}: {count:4d} ({pct:5.1f}%)")
        
        # Issue severity distribution
        severity_counts = defaultdict(int)
        total_issues = 0
        for ap in annotated_plans:
            total_issues += len(ap.annotation.issues)
            for issue in ap.annotation.issues:
                severity_counts[issue.get('severity', 'unknown')] += 1
        
        print(f"\nIssue Statistics:")
        print(f"  Total issues detected: {total_issues}")
        print(f"  Avg issues per plan: {total_issues / len(annotated_plans):.2f}")
        print(f"\n  By Severity:")
        for severity in ["critical", "high", "medium", "low", "unknown"]:
            count = severity_counts[severity]
            if count > 0:
                pct = 100 * count / total_issues if total_issues > 0 else 0
                print(f"    {severity:10s}: {count:4d} ({pct:5.1f}%)")
        
        # Verify rubric consistency
        print(f"\nRubric Consistency Check:")
        print(f"  (Verifying that error_type='none' has highest scores)")
        none_scores = error_type_scores.get(ErrorType.NONE, [])
        if none_scores:
            none_avg = sum(none_scores) / len(none_scores)
            error_scores = [s for et, scores in error_type_scores.items() 
                          if et != ErrorType.NONE for s in scores]
            if error_scores:
                error_avg = sum(error_scores) / len(error_scores)
                print(f"    Ground truth avg: {none_avg:.1f}")
                print(f"    Error plans avg:  {error_avg:.1f}")
                if none_avg > error_avg + 5:
                    print(f"    ✓ Rubric working: Ground truth scored {none_avg - error_avg:.1f} points higher")
                else:
                    print(f"    ⚠ Warning: Ground truth not scored significantly higher!")
        
        print(f"\n{'='*80}")


def main():
    """Main function with example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate candidate plans from ToolHop dataset")
    parser.add_argument("--toolhop-path", type=str, required=True, help="Path to ToolHop JSON file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save generated dataset")
    parser.add_argument("--api-key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Model to use for generation")
    parser.add_argument("--n-candidates", type=int, default=10, help="Number of candidates per query")
    parser.add_argument("--max-queries", type=int, default=None, help="Maximum number of queries to process")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation checks")
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(
        api_key=args.api_key,
        model=args.model
    )
    
    generator.generate_dataset(
        toolhop_path=args.toolhop_path,
        output_path=args.output_path,
        n_candidates_per_query=args.n_candidates,
        max_queries=args.max_queries,
        validate=not args.no_validate
    )


if __name__ == "__main__":
    main()