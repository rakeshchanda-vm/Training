Guardrails
==========

Safety controls and content filtering for AI agents.


Overview
--------

Guardrails provide:

* **Content Filtering**: Block inappropriate or harmful content
* **Safety Controls**: Prevent unsafe operations and outputs
* **Policy Enforcement**: Enforce organizational guidelines
* **Compliance**: Meet regulatory and security requirements

Middleware Integration
----------------------

Guardrails are implemented as middleware components:

.. code-block:: python

   from andromeda.core.agent import Agent
   from andromeda.config.config import AgentConfig, ModelConfig, MiddlewareConfig
   
   # Configure basic guardrails  
   cfg = AgentConfig(
       name="protected_agent",
       model=ModelConfig(name="gpt-4", provider="openai"),
       middleware=MiddlewareConfig(
           guardrails=MiddlewareConfig.GuardrailOptions(
               input=True,   # Block malicious inputs
               output=True,  # Filter unsafe outputs
               tool=False    # Skip tool result filtering
           )
       )
   )
   agent = Agent(cfg)
   
Refer to LangChain documentation for more details on middleware: https://docs.langchain.com/oss/python/langchain/middleware/overview

Prompt Injection Protection
---------------------------

Detect and block malicious prompt injection attempts:

.. code-block:: python

   from andromeda.core.agent import Agent
   from andromeda.config.config import (
       AgentConfig, 
       ModelConfig, 
       MiddlewareConfig,
       PromptInjectionPatternsConfig
   )

   # Configure prompt injection protection
   cfg = AgentConfig(
       name="secure_agent",
       model=ModelConfig(name="gpt-4", provider="openai"),
       middleware=MiddlewareConfig(
           guardrails=MiddlewareConfig.GuardrailOptions(
               input=True,  # Scan user inputs
               output=True,  # Check AI responses  
               tool=False,  # Skip tool results
               prompt_injection_patterns=PromptInjectionPatternsConfig(
                   patterns=[
                       # Blocks attempts to ignore previous instructions
                       r"ignore.*(previous|above|earlier)",
                       # Blocks attempts to make AI forget its rules
                       r"forget.*(instructions|rules)",
                       # Blocks role-switching attempts
                       r"act as.*different",
                       # Blocks impersonation requests
                       r"pretend.*you are"
                   ]
               ),
               blocked_message="I can't comply with prompt injection requests."
           )
       )
   )
   agent = Agent(cfg)

Advanced Configuration
----------------------

Tool Result Filtering
~~~~~~~~~~~~~~~~~~~~~~

Guardrails can also filter tool outputs for security violations:

.. code-block:: python

   cfg = AgentConfig(
       name="secure_agent",
       model=ModelConfig(name="gpt-4", provider="openai"),
       middleware=MiddlewareConfig(
           guardrails=MiddlewareConfig.GuardrailOptions(
               input=True,    # Check user inputs
               output=True,   # Validate agent responses
               tool=True,     # Also filter tool results  
               blocked_message="Security policy violation detected."
           )
       )
   )

Custom Blocked Messages
~~~~~~~~~~~~~~~~~~~~~~~

You can customize blocked messages for different scenarios:

.. code-block:: python

   # Global message for all guardrails violations
   GuardrailOptions(
       input=True,
       output=True,
       blocked_message="This request violates our security policy."
   )

Pattern Customization
~~~~~~~~~~~~~~~~~~~~~

**Override Default Patterns Completely:**

.. code-block:: python

   # Replace all default patterns with your own
   cfg = AgentConfig(
       middleware=MiddlewareConfig(
           guardrails=MiddlewareConfig.GuardrailOptions(
               input=True,
               prompt_injection_patterns=PromptInjectionPatternsConfig(
                   patterns=[
                       # Custom pattern for specific injection attempts
                       r"custom.*injection.*pattern",
                       # Additional security validation pattern
                       r"another.*security.*check"
                   ]
               ),
               compliance_patterns=CompliancePatternsConfig(
                   patterns=[
                       # Organization-specific compliance rule
                       r"custom.*compliance.*rule",
                       # Internal policy validation pattern
                       r"organization.*policy.*check"
                   ]
               )
           )
       )
   )

YAML Configuration
~~~~~~~~~~~~~~~~~~

Example YAML configuration for guardrails:

.. code-block:: yaml

   agents:
     - name: secure_agent
       model:
         name: gpt-4
         provider: openai
       middleware:
         enabled: true
         guardrails:
           input: true
           output: true
           tool: false
           blocked_message: "Security policy violation detected."
           prompt_injection_patterns:
             patterns:
               # Blocks attempts to override system instructions
               - "ignore\\s+all\\s+previous\\s+instructions"
               # Prevents privilege escalation attempts
               - "act\\s+as\\s+(?:admin|root|system)"
           compliance_patterns:
             patterns:
               # Prevents false guarantees in financial contexts
               - "guaranteed\\s+(?:approval|returns?)"
               # Blocks claims of absolute certainty
               - "cannot\\s+be\\s+denied"

Compliance Checking
-------------------

Ensure outputs meet regulatory and organizational standards:

.. code-block:: python

   from andromeda.config.config import CompliancePatternsConfig
   
   # Configure compliance guardrails
   cfg = AgentConfig(
       name="compliant_agent", 
       model=ModelConfig(name="claude-3", provider="anthropic"),
       middleware=MiddlewareConfig(
           guardrails=MiddlewareConfig.GuardrailOptions(
               output=True,
               tool=True,
               compliance_patterns=CompliancePatternsConfig(
                   patterns=[
                       # Prevents unauthorized medical advice
                       r"medical.*(advice|diagnosis|prescription)",
                       # Blocks investment recommendations
                       r"investment.*(advice|recommendation)",
                       # Prevents legal counsel without proper licensing
                       r"legal.*(advice|opinion)"
                   ]
               ),
               blocked_message="I can't provide professional advice in regulated areas."
           )
       )
   )
   agent = Agent(cfg)

Default Protection Patterns
---------------------------

Andromeda includes built-in protection patterns:

**Prompt Injection (complete default patterns):**
   * ``ignore\s+(all\s+)?previous\s+instructions``  # Prevents instruction override attempts
   * ``disregard\s+(the\s+)?system\s+prompt``  # Blocks system prompt manipulation
   * ``reveal\s+(the\s+)?system\s+prompt``  # Prevents system prompt extraction
   * ``bypass\s+(safety|guardrails|policy|security)``  # Stops security bypass attempts
   * ``disable\s+(safety|guardrails|policy|security)``  # Prevents safety feature disabling

**Compliance (complete default patterns):**
   * ``\bguaranteed\s+(approval|coverage|payout|returns?)\b``  # Prevents false financial guarantees
   * ``\bno\s+(exclusions|conditions|limitations)\b``  # Blocks misleading absolute claims
   * ``\bcannot\s+be\s+denied\b``  # Prevents false certainty statements
   * ``\bfalsif(?:y|ied|ication)\b``  # Detects document falsification attempts
   * ``\bmisrepresent(?:ation)?\b``  # Flags potential misrepresentation

Guardrails Examples
-------------------

Example on direct guardrail functionality :

.. code-block:: python

   from andromeda.core.middleware.guardrails import (
       PromptInjectionMiddleware,
       ComplianceMiddleware
   )

   # Test prompt injection detection
   middleware = PromptInjectionMiddleware(
       patterns=[
           # Pattern to catch instruction override attempts
           r"ignore.*instructions"
       ]
   )
   detected = middleware._contains_injection("ignore previous instructions")  # True

   # Test compliance checking
   compliance = ComplianceMiddleware(
       patterns=[
           # Pattern to detect unauthorized medical advice
           r"medical.*advice"
       ],
       replacement_message="I can't provide professional advice in regulated areas."
   )
   flagged = compliance._is_non_compliant("here's medical advice")  # True

   # Test with tool results
   privacy = DataPrivacyMiddleware(
       strategy="redact",
       apply_to_tool_results=True
   )
   
   def mock_handler(request):
       return ToolMessage(content="Contact: john@example.com", tool_call_id="test")
   
   result = privacy.wrap_tool_call(None, mock_handler)
   assert "[REDACTED_EMAIL]" in result.content

Best Practices
--------------

**Pattern Design**
   - Use specific patterns to avoid false positives
   - Test patterns with representative data before deployment
   - Avoid overly broad patterns like ``.*secret.*``

**Performance Considerations**
   - Complex regex patterns can impact response times
   - Pre-compile patterns for repeated use
   - Monitor for ReDoS (Regular Expression Denial of Service) vulnerabilities
   - Consider pattern execution order for efficiency

**Scope Configuration**
   - Use minimal scopes: apply guardrails only where needed
   - ``input=true`` for user input validation
   - ``output=true`` for response compliance checking  
   - ``tool=true`` only when tool results need filtering
   - Avoid enabling all scopes unless necessary

**Testing Strategy**
   - Test with edge cases and adversarial inputs
   - Validate patterns don't block legitimate content
   - Performance test with production-scale data
   - Document expected behavior for your patterns

Troubleshooting
---------------

**Common Issues:**

1. **Patterns Not Triggering**
   - Check regex syntax (remember to escape backslashes in YAML)
   - Test patterns independently with test strings
   - Verify scope settings match your use case

2. **False Positives**
   - Refine patterns to be more specific
   - Test with diverse legitimate inputs
   - Consider using compliance checking instead of blocking

3. **Performance Problems**
   - Profile pattern execution time
   - Simplify complex regex patterns
   - Reduce scope to minimize processing

**Debug Mode:**

.. code-block:: python

   # Enable debugging to see pattern matches
   import logging
   logging.getLogger("andromeda.core.middleware.guardrails").setLevel(logging.DEBUG)
   
   # Or enable full agent debugging
   agent_config = AgentConfig(debug=2, middleware=your_middleware_config)
