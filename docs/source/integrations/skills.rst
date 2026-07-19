Skills
======

Andromeda supports filesystem-based skills with progressive disclosure and runtime tool attachment.

Overview
--------

- Skills are discovered from configured source directories.
- Each skill must include a ``SKILL.md`` file with YAML frontmatter.
- Skills are exposed to the model by metadata first (name + description).
- Full skill behavior is activated by calling ``load_skill``.
- After a skill is loaded, tools listed in ``allowed-tools`` are resolved from the global Andromeda toolkit registry and attached for that run/thread.


Filesystem skill layout
-----------------------

.. code-block:: text

   skills/
     demo/
       sales-order/
         SKILL.md

Skill file format
-----------------

.. code-block:: markdown

   ---
   name: sales-order
   description: Handles sales order creation workflow.
   allowed-tools: create_order
   ---

   # Sales Order Skill
   Use this skill when users ask to create a sales order.

Middleware setup
----------------

``SkillsMiddleware`` now provides built-in backends and accepts a backend mode string:

.. code-block:: python

   from pathlib import Path
   from andromeda.core.middleware.skills import SkillsMiddleware

   skills_middleware = SkillsMiddleware(
       backend="filesystem",       # or "in-memory"
       repo_root=Path("."),        # used by filesystem backend
       sources=["/skills/demo"],   # virtual skill source paths
   )

For in-memory usage:

.. code-block:: python

   skills_middleware = SkillsMiddleware(
       backend="in-memory",
       in_memory_files={
           "/skills/demo/sales-order/SKILL.md": b"...",
       },
       sources=["/skills/demo"],
   )

Skill file format
-----------------

Example ``SKILL.md``:

.. code-block:: markdown

   ---
   name: sales-order
   description: Handles sales order creation workflow.
   allowed-tools: create_order
   ---

   # Sales Order Skill
   Use this skill when users ask to create a sales order.

How tools are attached
----------------------

1. Developers register tools globally via ``andromeda.tools.toolkit.register_tools``.
2. ``SkillsMiddleware`` discovers skills and injects built-in ``load_skill`` when skills exist.
3. ``load_skill`` updates middleware-managed state (``active_skills``).
4. On subsequent model calls, tools from ``allowed-tools`` for active skills are resolved from the toolkit and injected into the tool list.

Notes
-----

- You do not need to pass a backend instance anymore for common cases.
- ``load_skill`` is built into the middleware and auto-injected when skills are available.


Full example
------------

.. code-block:: python

    from pathlib import Path


    from andromeda import HumanMessage
    from andromeda.config import AgentConfig, MiddlewareConfig, ModelConfig
    from andromeda.core.agent import Agent
    from andromeda.core.middleware.skills import SkillsMiddleware
    from andromeda.tools import register_tools, tool


    @tool
    def create_order(customer_id: str, amount: float) -> str:
        """Create a sales order."""
        return f"Order created for customer={customer_id}, amount={amount:.2f}"


    def build_andromeda_agent() -> Agent:
        # Register tools once globally; skills middleware resolves from registry at runtime.
        register_tools([create_order])

        repo_root = Path(__file__).resolve().parent
        skills_middleware = SkillsMiddleware(
            backend="filesystem",
            repo_root=repo_root, # looks like "/app/src/"
            sources=["skills/demo"],
        )

        cfg = AgentConfig(
            name="skill-demo-agent",
            model=ModelConfig(
                name="gpt-oss:20b",
                provider="litellm",
                other_args={
                    "num_ctx": 4096 * 10,
                    "reasoning": True,
                },
            ),
            # No initial tools needed; middleware injects built-in load_skill when skills exist.
            tools=[],
            prompt=(
                "You are a sales operations agent. "
                "Load skills when needed, then use their allowed tools."
            ),
            middleware=MiddlewareConfig(
                enabled=True,
                custom=[skills_middleware],
            ),
        )
        return Agent(cfg)


    if __name__ == "__main__":
        agent = build_andromeda_agent()
        first_turn = agent.invoke([HumanMessage(content="Load the sales-order skill.")])
        second_turn = agent.invoke(
            [HumanMessage(content="Create an order for customer C-100 for 249.99.")]
        )

        print("=== First turn ===")
        for m in first_turn:
            print(f"{m.type}: {getattr(m, 'content', m)}")

        print("\n=== Second turn ===")
        for m in second_turn:
            print(f"{m.type}: {getattr(m, 'content', m)}")
