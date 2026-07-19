Debugging Workflows
===================

Troubleshooting and error diagnosis for agent workflows with comprehensive debugging tools.

Overview
--------

Andromeda provides multiple debugging capabilities:

* **Agent Debug Levels**: 4-level debugging system (0-3) with increasing detail
* **Color-Coded Logging**: Structured logging with visual categorization  
* **Method Tracing**: Automatic tracing with PyEZTrace integration
* **Workflow Debugging**: Step-by-step workflow execution monitoring
* **Error Routing**: Built-in error handling with debug paths

Agent Debug Levels
------------------

Enable different debug levels in your agent configuration:

.. code-block:: python

   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig
   from langchain.messages import HumanMessage

   # Debug Level 0: No debugging (default)
   config_level_0 = AgentConfig(
       name="normal_agent",
       model=ModelConfig(name="llama3:8b", provider="litellm"),
       debug=0  # No debug output
   )

   # Debug Level 1: Input/Output logging only
   config_level_1 = AgentConfig(
       name="io_debug_agent", 
       model=ModelConfig(name="llama3:8b", provider="litellm"),
       debug=1  # Shows input and output messages
   )

   # Debug Level 2: Full method tracing
   config_level_2 = AgentConfig(
       name="trace_agent",
       model=ModelConfig(name="llama3:8b", provider="litellm"), 
       debug=2  # Traces all method calls with PyEZTrace
   )

   # Debug Level 3: Full tracing + LangGraph debugging
   config_level_3 = AgentConfig(
       name="full_debug_agent",
       model=ModelConfig(name="llama3:8b", provider="litellm"),
       debug=3  # Maximum debugging with LangGraph internals
   )

   # Create and use debug agent
   debug_agent = Agent(config_level_1)
   messages = [HumanMessage(content="Hello, debug me!")]
   result = debug_agent.invoke(messages)  # Will show I/O logs

Color-Coded Logging System
--------------------------

Use structured logging with color-coded output:

.. code-block:: python

   from andromeda.utils.logger import (
       log_supervisor, log_agent, log_tool, 
       log_input, log_output, log_error, log_warning
   )

   # Different log types with colors
   log_supervisor("Starting workflow execution")        # Blue
   log_agent("my_agent", "Processing user request")     # Black  
   log_tool("web_search", "Searching for information")  # Yellow
   log_input("User asked: What is debugging?")          # Cyan
   log_output("Agent responded with helpful info")      # Magenta
   log_warning("This operation might take time")        # Yellow
   log_error("Something went wrong!")                   # Red

   # Pretty printing for complex outputs
   debug_data = {
       "agent_state": "active",
       "tools_available": ["search", "calculator"],
       "current_step": "reasoning"
   }
   log_output(debug_data, pretty=True)  # Formatted output

Workflow Debugging
------------------

Debug workflow execution with monitoring:

.. code-block:: python

   from typing import Dict, Any
   from andromeda.core.workflow import WorkflowBuilder
   from andromeda.utils.logger import log_agent

   # Define workflow steps with debug logging
   def debug_step_one(state: Dict[str, Any]) -> Dict[str, Any]:
       log_agent("workflow", f"Step 1 - Input state: {state}")
       result = {"step1_completed": True, "data": state.get("input", "")}
       log_agent("workflow", f"Step 1 - Output: {result}")
       return result

   def debug_step_two(state: Dict[str, Any]) -> Dict[str, Any]:
       log_agent("workflow", f"Step 2 - Input state: {state}")
       result = {"step2_completed": True, "processed": state.get("data", "") + "_processed"}
       log_agent("workflow", f"Step 2 - Output: {result}")
       return result

   # Create workflow with debugging
   workflow = WorkflowBuilder(name="DebugWorkflow")
   (
       workflow
       .start("step_one").run(debug_step_one)
       .finish("step_two").run(debug_step_two)
   )

   # Execute with debug monitoring
   result = workflow.execute(
       state={"input": "test_data"},
       debug=True,      # Enable workflow debugging
       monitor=True     # Enable execution monitoring
   )

   # Stream workflow for step-by-step debugging  
   print("\\nStreaming workflow execution:")
   for chunk in workflow.stream(state={"input": "stream_test"}):
       print(f"Debug chunk: {chunk}")

Error Handling and Debug Logging
---------------------------------

Debug workflow execution with comprehensive error logging:

.. code-block:: python

   from andromeda.core.workflow import WorkflowBuilder
   from andromeda.utils.logger import log_error, log_warning, log_agent

   def risky_operation(state: Dict[str, Any]) -> Dict[str, Any]:
       """Operation that might fail for debugging demonstration."""
       log_agent("risky_operation", f"Processing: {state}")
       
       if state.get("should_fail", False):
           log_error("Simulated failure in risky operation")
           return {"error": "Intentional failure", "failed": True}
       
       log_agent("risky_operation", "Operation succeeded")
       return {"success": True, "data": "operation completed"}

   def process_result(state: Dict[str, Any]) -> Dict[str, Any]:
       """Handle both success and error cases."""
       if state.get("error"):
           log_warning(f"Handling error: {state['error']}")
           return {
               "error_handled": True,
               "recovery_action": "logged_and_documented",
               "final_status": "completed_with_errors"
           }
       else:
           log_agent("process_result", "Processing successful result")
           return {"final_status": "completed_successfully", "data": state}

   # Build simple workflow for debugging
   debug_workflow = WorkflowBuilder(name="DebugErrorWorkflow")
   (
       debug_workflow
       .start("risky_step")
           .run(risky_operation)
       .then("process")
           .run(process_result)
       .finish("complete")
   )

   # Test with failure (handled gracefully)
   print("Testing with failure (handled gracefully):")
   fail_result = debug_workflow.execute(state={"should_fail": True})
   print(f"Failure result: {fail_result}")

   # Test with success  
   print("\\nTesting with success:")
   success_result = debug_workflow.execute(state={"should_fail": False})
   print(f"Success result: {success_result}")

Method Tracing with PyEZTrace
-----------------------------

Enable automatic method tracing for deep debugging:

.. code-block:: python

   from pyeztrace.tracer import trace
   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig

   # Manual tracing decorator
   @trace(exclude=["log_*"])  # Exclude log functions from traces
   def custom_debug_function(data):
       log_agent("tracer", f"Processing: {data}")
       return {"processed": data, "timestamp": "traced"}

   # Agent with automatic tracing (debug=2)
   traced_agent = Agent(AgentConfig(
       name="traced_agent",
       model=ModelConfig(name="llama3:8b", provider="litellm"),
       debug=2  # Enables automatic tracing of all agent methods
   ))

   # All these methods will be automatically traced:
   # - invoke() / ainvoke()
   # - stream() / astream()  
   # - chat() / achat()
   # - task() / atask()
   # - research() / aresearch()

Performance Monitoring
----------------------

Monitor workflow execution and identify bottlenecks:

.. code-block:: python

   import time
   from andromeda.core.workflow import WorkflowBuilder
   from andromeda.utils.logger import log_agent

   def fast_step(state: Dict[str, Any]) -> Dict[str, Any]:
       start_time = time.time()
       result = {"fast_data": "quick_result"}
       duration = time.time() - start_time
       log_agent("performance", f"Fast step completed in {duration:.4f}s")
       return result

   def slow_step(state: Dict[str, Any]) -> Dict[str, Any]:
       start_time = time.time()
       time.sleep(0.1)  # Simulate slow operation
       result = {"slow_data": "delayed_result"}
       duration = time.time() - start_time
       log_agent("performance", f"Slow step completed in {duration:.4f}s")
       return result

   # Performance monitoring workflow
   perf_workflow = WorkflowBuilder(name="PerformanceWorkflow")
   (
       perf_workflow
       .start("fast").run(fast_step)
       .finish("slow").run(slow_step)
   )

   # Execute with timing
   start_total = time.time()
   result = perf_workflow.execute(
       state={}, 
       monitor=True  # Enable performance monitoring
   )
   total_duration = time.time() - start_total
   log_agent("performance", f"Total workflow time: {total_duration:.4f}s")

CLI Debugging Tools
-------------------

Andromeda provides command-line debugging utilities:

**Diagnostic Command:**

.. code-block:: bash

   # Run comprehensive diagnostic checks
   andromeda diagnose
   
   # Check for common configuration issues, dependencies, and environment setup

**Debug Environment Variables:**

.. code-block:: bash

   # Enable debug mode for CLI operations
   export DEBUG=1
   
   # Configure PyEZTrace logging levels
   export EZTRACE_LOG_LEVEL=DEBUG  # Options: DEBUG, INFO, WARNING, ERROR

**Generated Configs with Debug:**

The CLI automatically generates configurations with debug level 2 enabled:

.. code-block:: bash

   # Generated templates include debug configuration
   andromeda new my-project
   # Creates config with debug=2 for development

Advanced Debugging Features
---------------------------

**Async Method Tracing:**

All async agent methods are automatically traced when debug=2:

.. code-block:: python

   traced_agent = Agent(AgentConfig(debug=2))
   
   # All these async methods will be traced:
   await traced_agent.ainvoke(messages)
   await traced_agent.astream(messages)
   await traced_agent.achat("Hello")
   await traced_agent.atask("Complete this task")
   await traced_agent.aresearch("Research this topic")
   async for event in traced_agent.astream_structured_events(messages):
       print(f"Debug event: {event}")

**Environment Variable Setup:**

.. code-block:: bash

   # Complete debug environment setup
   export DEBUG=1                          # CLI debug mode
   export EZTRACE_LOG_LEVEL=DEBUG         # PyEZTrace logging
   export LANGFUSE_SECRET_KEY=your_key    # Optional: Langfuse tracing
   export LANGFUSE_PUBLIC_KEY=your_key    # Optional: Langfuse tracing  
   export LANGFUSE_HOST=your_host         # Optional: Langfuse tracing

**Thread ID and Metadata Debugging:**

.. code-block:: python

   # Set debugging context for workflow tracking
   agent.set_thread_id("debug-session-001")
   agent.set_metadata({"debug": True, "session": "test"})
   
   # Execute with tracking
   result = agent.invoke(messages)  # Includes context in logs

Middleware Debugging
-------------------

Debug middleware components including ComplianceMiddleware for pattern matching and compliance checking:

**ComplianceMiddleware Debugging:**

.. code-block:: python

   from andromeda.core.middleware.guardrails import ComplianceMiddleware
   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, MiddlewareConfig
   
   # Enable ComplianceMiddleware with debug patterns
   compliance = ComplianceMiddleware(
       apply_to_output=True,        # Debug AI responses
       apply_to_tool_results=True,  # Debug tool outputs
       patterns=[
           r"medical.*advice",       # Catches "medical advice", "medical treatment advice"
           r"guaranteed.*cure",      # Catches "guaranteed cure", "guaranteed to cure"
           r"no.*side.*effects",     # Catches "no side effects", "no harmful side effects"
           r"definitely.*will.*work" # Catches "definitely will work", "definitely this will work"
       ],
       replacement_message="[COMPLIANCE BLOCK] Content violates compliance policies."
   )
   
   # Test compliance pattern matching
   test_inputs = [
       "This treatment provides medical advice for everyone.",     # ✗ BLOCKED
       "We guarantee this cure will work perfectly.",            # ✗ BLOCKED  
       "There are absolutely no side effects at all.",          # ✗ BLOCKED
       "This definitely will work for your condition.",         # ✗ BLOCKED
       "This is general health information only."               # ✅ ALLOWED
   ]
   
   for text in test_inputs:
       is_compliant = not compliance._is_non_compliant(text)
       print(f"Text: {text}")
       print(f"Compliant: {is_compliant}")
       print("---")

**ComplianceMiddleware Capabilities:**

The ComplianceMiddleware can perform the following debugging and compliance operations:

- **Pattern Detection**: Uses regex patterns to detect non-compliant content
- **Output Filtering**: Scans AI responses for compliance violations (``apply_to_output=True``)  
- **Tool Result Filtering**: Optionally scans tool outputs (``apply_to_tool_results=True``)
- **Content Replacement**: Replaces flagged content with configurable messages
- **Case-Insensitive Matching**: Patterns use ``re.IGNORECASE`` for robust detection
- **Configurable Patterns**: Supports custom regex patterns for specific compliance needs

**Built-in Compliance Patterns:**

.. code-block:: python

   # Default patterns from CompliancePatternsConfig
   default_patterns = [
       r"\bguaranteed\s+(approval|coverage|payout|returns?)\b",  # "guaranteed approval"
       r"\bno\s+(exclusions|conditions|limitations)\b",          # "no exclusions"  
       r"\bcannot\s+be\s+denied\b",                              # "cannot be denied"
       r"\bfalsif(?:y|ied|ication)\b",                           # "falsify", "falsification"
       r"\bmisrepresent(?:ation)?\b"                            # "misrepresent", "misrepresentation"
   ]

**Middleware Integration with Agent Debugging:**

.. code-block:: python

   # Combine middleware debugging with agent debug levels
   debug_config = AgentConfig(
       name="compliance_debug_agent",
       model=ModelConfig(name="gpt-4", provider="openai"),
       debug=2,  # Enable method tracing
       middleware=MiddlewareConfig(
           guardrails=MiddlewareConfig.GuardrailOptions(
               output=True,
               compliance_patterns=CompliancePatternsConfig(
                   patterns=[
                       r"medical.*advice",
                       r"legal.*recommendation",
                       r"financial.*guarantee"
                   ]
               )
           )
       )
   )
   
   debug_agent = Agent(debug_config)
   # Will trace method calls AND apply compliance filtering

Troubleshooting Guide
--------------------

**Common Debug Issues:**

1. **PyEZTrace Not Working**
   - Verify installation: ``pip install pyeztrace>=0.0.7``
   - Check environment: ``EZTRACE_LOG_LEVEL=DEBUG``
   - Ensure debug level 2: ``debug=2`` in AgentConfig

2. **Missing CLI Diagnostics**
   - Update to latest version
   - Run: ``andromeda diagnose`` for system health check
   - Use specific flags: ``--check-deps``, ``--test-connections``, ``--check-env``

CLI Debugging Tools
-------------------

**System Diagnostics Command:**

The ``andromeda diagnose`` command provides comprehensive system debugging:

.. code-block:: bash

   # Run complete diagnostic suite
   andromeda diagnose
   
   # Check specific components
   andromeda diagnose --check-deps        # Dependencies only
   andromeda diagnose --test-connections  # External services only  
   andromeda diagnose --check-env         # Environment variables only
   andromeda diagnose --check-config      # Configuration validation only
   andromeda diagnose --check-tools       # Tool registry validation only

**Diagnostic Components:**

1. **Dependency Checking**
   - Python package versions and compatibility
   - System command availability (pip, git)
   - Version requirement validation
   - Installation status reporting

2. **Service Connection Testing**
   - Ollama server (localhost:11434) - TCP socket test
   - Tavily API (api.tavily.com) - HTTPS connectivity
   - Custom service endpoint validation
   - Timeout handling and error reporting

3. **Environment Variable Validation**
   - Optional API keys detection
   - Configuration completeness check
   - Missing variable identification
   - .env file generation suggestions

4. **Configuration File Validation** 
   - YAML syntax and structure validation
   - AndromedaConfig parsing verification
   - Schema compliance checking
   - Configuration integrity reporting

5. **Tool Registry Inspection**
   - Tool registration verification
   - Agent-tool binding validation
   - Configuration mismatch detection
   - Tool availability reporting

**Debug Environment Variables:**

.. code-block:: bash

   # CLI debugging mode
   export DEBUG=1                     # Enable CLI error tracebacks
   
   # PyEZTrace debugging
   export EZTRACE_LOG_LEVEL=DEBUG    # Detailed tracing logs
   
   # Generate environment with debug features
   andromeda generate-env --include-pyeztrace
   
**CLI Error Handling:**

When ``DEBUG=1`` is set, the CLI provides detailed error tracebacks:

.. code-block:: bash

   # Enable detailed error reporting
   export DEBUG=1
   andromeda run my_config.yaml  # Shows full tracebacks on errors

**Environment Generation with Debugging:**

.. code-block:: bash

   # Create debug-ready .env file
   andromeda generate-env --include-pyeztrace --include-optional
   
   # Interactive mode with all options
   andromeda generate-env --interactive
   
**Quick Debugging Workflow:**

.. code-block:: bash

   # 1. Check system health
   andromeda diagnose
   
   # 2. Enable debug mode  
   export DEBUG=1
   export EZTRACE_LOG_LEVEL=DEBUG
   
   # 3. Validate specific configuration
   andromeda diagnose --check-config
   
   # 4. Test run with debugging
   andromeda run config.yaml
   - Check installation: ``andromeda diagnose``
   - Verify PATH includes andromeda command

3. **Workflow Debug Not Showing**
   - Enable both: ``debug=True`` and ``monitor=True``
   - Check log levels in environment
   - Verify workflow steps include logging

4. **LangGraph Debug Silent**
   - Ensure debug level 3: ``debug=3``
   - Check LangGraph installation
   - Verify agent type supports debugging (CodeAct vs ReAct)

**Debug Command Reference:**

.. code-block:: bash

   # Quick debugging checklist
   andromeda diagnose                    # System diagnostics
   python -c "import pyeztrace; print('OK')"  # Check PyEZTrace
   echo $DEBUG $EZTRACE_LOG_LEVEL       # Check environment
   
   # Test agent debugging
   python -c "
   from andromeda import Agent, AgentConfig, ModelConfig
   agent = Agent(AgentConfig(name='test', model='gpt-4', debug=2))
   print('Debug agent created successfully')
   "
