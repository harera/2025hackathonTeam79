import json
import asyncio
from datetime import datetime
from typing import Dict, Any

from azure.identity import DefaultAzureCredential
from azure.ai.projects.aio import AIProjectClient
from semantic_kernel.agents import AzureAIAgent
import logging

logger = logging.getLogger(__name__)

from autogen import ConversableAgent

# --- 代理 Foundry 中的决策 Agent ---
class FoundryDecisionAgent:
    def __init__(self):
        self.client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str="eastus2.api.azureml.ms;0dd70ca3-209c-42b7-8dbe-2de878b7b127;odl-sandbox-1660037-02;aironwomen",
        )

    async def call_decision_agent(self, input: str) -> str:
        """
        Call Foundry Agent for loan decision, based on previous agent outputs
        """
        try:
            agent_definition = await self.client.agents.get_agent(agent_id="asst_pn3mmhpcgLJjJC14xAmLGqvC")
            agent = AzureAIAgent(client=self.client, definition=agent_definition)
            response = await agent.get_response(messages=input, thread=None)
            
            # 处理不同类型的响应
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                return response.message.content
            elif hasattr(response, 'content'):
                return str(response.content)
            elif isinstance(response, str):
                return response
            else:
                logger.error(f"Unexpected response type: {type(response)}")
                return "No valid response content found"
        except Exception as e:
            logger.error(f"Error calling Foundry decision agent: {str(e)}")
            return f"[ERROR calling Foundry decision agent]: {str(e)}"
        finally:
            if hasattr(self, 'client'):
                await self.client.close()

# --- AutoGen Agent 封装 ---
class LoanDecisionAgent(ConversableAgent):
    def __init__(self, name="LoanDecisionAgent"):
        llm_config = {
            "config_list": [{
                "model": "gpt-4o-mini",
                "api_key": "U4fD1XnytGhxvUlsRYSRL472aImrcK17LbNS5CQfGOTRpkWKAwpzJQQJ99BDACHYHv6XJ3w3AAAAACOGIngS",
                "base_url": "https://hackathonfound6454649233.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview"
            }]
        }

        super().__init__(
            name=name,
            system_message="你是一个房贷决策 AI，负责汇总其他 Agent 分析结果，并调用 Foundry 决策模型。",
            human_input_mode="NEVER",
            llm_config=llm_config
        )

        async def make_loan_decision(summary_input: str) -> Dict[str, Any]:
            """调用 Foundry 决策 Agent 获取结果"""
            foundry = FoundryDecisionAgent()
            decision = await foundry.call_decision_agent(summary_input)
            return {
                "status": "success",
                "decision_result": decision,
                "timestamp": datetime.now().isoformat()
            }

        self.make_loan_decision = make_loan_decision
        self.register_function({"make_loan_decision": make_loan_decision})

# --- 示例调用 ---
async def demo():
    # 示例输入：其他 Agent 总结输出
    summary_input = """
    信用评估：用户信用等级为 A，信用分为 720；
    欺诈检测：无明显欺诈行为；
    合规检查：符合本行房贷申请标准。
        """

    agent = LoanDecisionAgent()
    response = await agent.make_loan_decision(json.dumps(summary_input))
    print(json.dumps(response, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(demo())
