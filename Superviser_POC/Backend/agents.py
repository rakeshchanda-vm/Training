from tools import step_memo, step_report,get_schema,run_query,run_process_file
from andromeda.config import AgentConfig, ModelConfig, SupervisorConfig
from andromeda import HumanMessage
from config import get_agent_tools, get_genai_llm, PROVIDER, GEN_MODEL
from andromeda.core.supervisor import Supervisor


################ AGENTS ############################

Agent_Summarize = get_agent_tools(name = "Summarize Agent",
                                prompt = "You are a Summarizer agent helping in summarize the complete data. Call this agent when you want summary or detailed analytical report",
                                tools = [step_memo,step_report])

Agent_SQL = get_agent_tools(name = "SQL Agent",
                                prompt = """You are a SQL agent helping to retriver relvant data. 
                                Using data present in Sqlite DB you can help answering any queries by hitting the DB. Always checks scheama befre writing queries""",
                                tools = [get_schema,run_query])

Agent_Process_File = get_agent_tools(name = "Processing Agent",
                                prompt = """You are a File processing agent helping to process file. If new files comes, this agent helps to structure data before summarizing or using for any analysis""",
                                tools = [run_process_file])


summarize_cfg = SupervisorConfig(
    name="supervisor",
    model=ModelConfig(name=GEN_MODEL,provider=PROVIDER, temperature=0),
    prompt="Plan and coordinate agents to fully cover the task. Use only the provided agents for task delegation. Always check agent descriptions to get idea which agent to call. Dont respond directly or dont use your prior knowledge",
    enable_planning=True,
)

Agent_Summarize = Supervisor(agents =[Agent_Summarize,Agent_Process_File,Agent_SQL],config = summarize_cfg) 


