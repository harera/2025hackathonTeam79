import os
import sys
import json
import logging
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv
import openai
import autogen
from concurrent.futures import ThreadPoolExecutor
from credit_agent import CreditAnalysisAgent, credit_analysis_workflow  # 导入CreditAnalysisAgent和credit_analysis_workflow
from fraud_agent import FoundryIncomeAgent, fraud_analysis_workflow  # 导入fraud_analysis_workflow
from decision_agent import LoanDecisionAgent  # 导入LoanDecisionAgent
from datetime import datetime
from compliance_agent import ComplianceReview, compliance_review_workflow
import uuid
import argparse
import io

# 设置控制台输出编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 创建日志目录
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 生成日志文件名
log_filename = os.path.join(log_dir, f"loan_application_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# 配置根日志记录器
logging.basicConfig(
    level=logging.INFO,  # 默认日志级别设为 INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# 设置特定模块的日志级别
logging.getLogger('credit_evaluator').setLevel(logging.INFO)
logging.getLogger('fraud_evaluator').setLevel(logging.INFO)
logging.getLogger('httpcore').setLevel(logging.WARNING)  # 减少 HTTP 相关的日志输出
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('docker').setLevel(logging.WARNING)

# 获取主应用程序的日志记录器
logger = logging.getLogger('loan_application')

# 加载环境变量
load_dotenv()

# 确保OpenAI库使用正确的配置
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# 配置autogen的模型
config_list = [
    {
        "model": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "api_type": "azure",
    }
]

# 定义llm_config
llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
    "functions": []  # 移除预定义的函数列表，改为在注册时动态添加
}

# 配置Azure存储
azure_storage_config = {
    "connection_string": os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    "container_name": os.getenv("AZURE_STORAGE_CONTAINER_NAME"),
    "account_name": os.getenv("AZURE_STORAGE_ACCOUNT_NAME"),
    "account_key": os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
}

# 配置Cosmos DB
cosmos_config = {
    "endpoint": os.getenv("COSMOS_ENDPOINT"),
    "key": os.getenv("COSMOS_KEY"),
    "database_name": os.getenv("COSMOS_DATABASE_NAME"),
    "container_name": os.getenv("COSMOS_CONTAINER_NAME"),
    "chat_container_name": os.getenv("COSMOS_CHAT_CONTAINER_NAME")
}

class LoanApplication:
    def __init__(self):
        """初始化贷款申请系统"""
        self.instance_id = str(uuid.uuid4())
        self._configure_logging()
        
        # 初始化分析结果
        self.credit_result = None
        self.fraud_result = None
        self.compliance_result = None
        self.decision_result = None
        
        # 初始化决策代理
        self.decision_agent = LoanDecisionAgent()
        
        # 初始化会话和连接器列表
        self._sessions = []
        self._connectors = []

    async def run(self):
        """运行贷款申请流程"""
        try:
            # 第一阶段：数据收集
            print("\n第一阶段: 数据收集")
            success = await self.data_collection()
            
            # 第二阶段：贷款评估
            if success and self.collected_data:
                print("\n第二阶段: 贷款评估")
                await self.phase_evaluation()
                
        except Exception as e:
            logging.error(f"贷款申请流程出错: {str(e)}")
            return {"status": "error", "error_message": str(e)}
            
        return {"status": "completed", "application_id": self.collected_data.get("application_id")}
    
    async def data_collection(self):
        """收集贷款申请数据"""
        try:
            print("\n=== 数据收集阶段 ===")
            
            # 创建数据收集agent
            data_collector = autogen.AssistantAgent(
                name="DataCollector",
                system_message="""你是数据收集专家，负责:
                    1. 收集贷款申请所需的所有信息
                    2. 确保数据的完整性和准确性
                    3. 验证数据的有效性""",
                llm_config={
                    "config_list": config_list,
                    "functions": []
                }
            )
            
            # 创建用户代理
            user_proxy = autogen.UserProxyAgent(
                name="User",
                system_message="你代表贷款申请人，负责提供申请所需的信息。",
                human_input_mode="ALWAYS",
                code_execution_config=False
            )
            
            # 启动数据收集对话
            chat_result = await user_proxy.initiate_chat(
                data_collector,
                message="请帮助我收集贷款申请所需的信息。"
            )
            
            # # 提取收集到的数据
            # for message in chat_result.chat_history:
            #     if "DATA_COLLECTION_COMPLETE" in message.get("content", ""):
            #         # 解析数据
            #         data = self.parse_collected_data(message.get("content", ""))
            #         if data:
            #             self.collected_data = data
            #             return True
            
            # return False
            
        except Exception as e:
            logging.error(f"数据收集过程出错: {str(e)}")
            return False
    
    def parse_collected_data(self, summary):
        """从agent的最终摘要中提取结构化数据"""
        # 提取"收集的数据摘要:"部分
        if "收集的数据摘要:" in summary:
            # 从收集的数据摘要:开始截取
            data_section = summary.split("收集的数据摘要:")[1]
            
            # 如果有DATA_COLLECTION_COMPLETE标记，就截止到该标记
            if "DATA_COLLECTION_COMPLETE" in data_section:
                data_section = data_section.split("DATA_COLLECTION_COMPLETE")[0]
            
            # 如果有"是否同意"类型的文本，也截止到此
            if "是否同意" in data_section:
                data_section = data_section.split("是否同意")[0]
                
            data_section = data_section.strip()
            
            # 将数据解析到字典中
            data_dict = {}
            for line in data_section.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    try:
                        key, value = line[2:].split(":", 1)
                        data_dict[key.strip()] = value.strip()
                    except ValueError:
                        continue
            
            return data_dict
        else:
            # 如果格式不符合预期则使用备用方案
            return {"summary": summary}
    
    async def phase_evaluation(self, test_data=None, session=None):
        """贷款评估阶段"""
        try:
            self.logger.info("开始贷款评估阶段")
            
            # 如果有测试数据，使用测试数据
            if test_data:
                self.collected_data = test_data
                self.logger.info("使用测试数据进行评估")
            
            # 创建专家代理
            agents = [
                DecisionAgent(
                    name="DecisionAgent",
                    llm_config=llm_config
                ),
                CreditExpert(
                    name="CreditExpert", 
                    llm_config=llm_config
                ),
                FraudExpert(
                    name="FraudExpert", 
                    llm_config=llm_config
                ),
                ComplianceExpert(
                    name="ComplianceExpert", 
                    llm_config=llm_config
                )
            ]
            self.logger.info("专家代理创建完成")
            
            # 为每个代理注册函数
            for agent in agents:
                register_functions(agent)
            
            # 创建用户代理
            user_proxy = autogen.UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=10,
                is_termination_msg=lambda x: "EVALUATION_COMPLETE" in x.get("content", ""),
                code_execution_config={"work_dir": "workspace"},
                llm_config=llm_config,
                system_message="""你是一个贷款申请人的代理。你的职责是：
                1. 提供申请人的详细信息
                2. 回答评估过程中的问题
                3. 配合完成整个评估流程
                4. 确保所有必要信息都已提供"""
            )
            
            # 创建群组聊天
            groupchat = autogen.GroupChat(
                agents=[user_proxy] + agents,
                messages=[],
                max_round=15
            )
            
            # 创建群组聊天管理器
            chat_manager = autogen.GroupChatManager(
                groupchat=groupchat,
                llm_config=llm_config
            )
            
            # 获取评估数据
            evaluation_data = await self.get_evaluation_data()
            
            # 执行信用分析
            credit_result = await self.run_credit_analysis(
                certificate_file=evaluation_data['证明文件']['工作证明'],
                bank_statement_file=evaluation_data['证明文件']['银行流水']
            )
            
            # 执行欺诈分析
            fraud_result = await self.run_fraud_analysis(
                evaluation_data['证明文件']['用户ID']
            )
            
            # 执行合规分析
            compliance_result = await self.run_compliance_analysis(
                evaluation_data['证明文件']['用户ID']
            )

            # 执行决策分析
            decision_result = await self.run_decision_analysis(
                evaluation_data['证明文件']['用户ID']
            )
            
            # 启动群聊评估并获取结果
            self.logger.info("开始群组聊天评估")
            chat_result = await chat_manager.a_initiate_chat(
                agents[1],  # credit_expert
                message=f"""
            
请评估以下贷款申请：

申请人信息：
{json.dumps(evaluation_data['申请人信息'], ensure_ascii=False, indent=2)}

工作信息：
{json.dumps(evaluation_data['工作信息'], ensure_ascii=False, indent=2)}

贷款信息：
{json.dumps(evaluation_data['贷款信息'], ensure_ascii=False, indent=2)}

证明文件：
{json.dumps(evaluation_data['证明文件'], ensure_ascii=False, indent=2)}

信用分析结果：
{json.dumps(credit_result, ensure_ascii=False, indent=2)}

欺诈分析结果：
{json.dumps(fraud_result, ensure_ascii=False, indent=2)}

合规分析结果：
{json.dumps(compliance_result, ensure_ascii=False, indent=2)}

决策结果：
{json.dumps(decision_result, ensure_ascii=False, indent=2)}

请开始评估。
"""
            )
            self.logger.info(f"群组聊天完成，聊天历史长度: {len(chat_result.chat_history) if chat_result and hasattr(chat_result, 'chat_history') else 0}")

            # 收集agent消息
            self.logger.info("开始收集agent消息")
            agent_messages = self.collect_expert_messages(chat_result.chat_history)
            self.logger.info(f"收集到 {len(agent_messages)} 条agent消息")
            
            # 构建评估结果
            evaluation_result = {
                "status": "completed",
                "application_id": self.collected_data.get("application_id"),
                "agent_messages": agent_messages
            }
            self.logger.info("评估结果构建完成")

            # 存入会话
            if session:
                self.logger.info("开始将消息存入会话")
                if "chat_history" not in session:
                    session["chat_history"] = []
                    self.logger.info("创建新的chat_history列表")
                
                for msg in evaluation_result['agent_messages']:
                    session["chat_history"].append({
                        "role": msg['name'],
                        "content": msg['content']
                    })
                self.logger.info(f"会话存储完成，当前chat_history长度: {len(session['chat_history'])}")
            else:
                self.logger.warning("session对象不存在，无法存储会话历史")
            
            print("\n=== 会话履历 ===")
            print(evaluation_result)
            return evaluation_result

        except Exception as e:
            self.logger.error(f"贷款评估失败: {str(e)}", exc_info=True)
            raise

    # 添加测试数据生成功能，方便调试    
    def generate_test_data(self):
        """生成测试数据，方便调试"""
        self.collected_data = {
            "姓名": "张三",
            "年龄": "30",
            "电话": "1234567890",
            "电子邮件": "zhangsan@example.com",
            "地址": "北京市海淀区",
            "工作单位": "索尔维亚首都银行",
            "职务": "软件工程师",
            "月收入": "30万元",
            "贷款金额": "50万元",
            "贷款用途": "购买房产",
            "贷款期限": "20年",
            "房屋地址": "北京市海淀区",
            "贷款开始日期": "2024年1月1日",
            "房屋面积": "120平方米",
            "employment_certificate": "yinhangliushui.png",  # 修正文件名
            "bank_statement": "zhangsanzaizhi.png",  # 修正文件名
            "user_id": "08c6c7e1-023d-4d21-a3d2-bb2d32efc7ca",
            "application_id": "08c6c7e1-023d-4d21-a3d2-bb2d32efc7ca",
            "session": "08c6c7e1-023d-4d21-a3d2-bb2d32efc7ca",
            "chat_history": []
        }
        print("已生成测试数据:")
        for key, value in self.collected_data.items():
            print(f"- {key}: {value}")
        return self.collected_data

    async def run_initial_analysis(self, loan_data):
        """执行初始贷款分析"""
        try:
            print("\n=== 执行初始分析 ===")
            print(f"分析数据: {loan_data}")
            
            # 执行分析
            result = await self.decision_agent.make_loan_decision(loan_data)
            
            if isinstance(result, dict) and result.get('status') == 'success':
                return f"初始分析结果: {result.get('decision_result', '无结果')}"
            else:
                return f"初始分析失败: {str(result)}"
                
        except Exception as e:
            logging.error(f"初始分析过程出错: {str(e)}")
            return f"初始分析失败: {str(e)}"

    async def run_credit_analysis(self, certificate_file: str, bank_statement_file: str) -> Dict[str, Any]:
        """执行信用分析"""
        try:
            self.logger.info("开始信用分析")
            result = await credit_analysis_workflow(certificate_file, bank_statement_file)
            self.credit_result = result
            return result
        except Exception as e:
            self.logger.error(f"信用分析失败: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def run_fraud_analysis(self, user_id: str) -> Dict[str, Any]:
        """执行欺诈分析"""
        try:
            self.logger.info("开始欺诈分析")
            result = await fraud_analysis_workflow(user_id)
            self.fraud_result = result
            return result
        except Exception as e:
            self.logger.error(f"欺诈分析失败: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def run_compliance_analysis(self, user_id: str) -> Dict[str, Any]:
        """执行合规分析"""
        try:
            self.logger.info("开始合规分析")
            result = await compliance_review_workflow(user_id)
            self.compliance_result = result
            return result
        except Exception as e:
            self.logger.error(f"合规分析失败: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def run_decision_analysis(self, user_id: str) -> Dict[str, Any]:
        """执行决策分析"""
        try:
            self.logger.info("开始决策分析")
            
            # 检查所有分析结果是否就绪
            if not all([self.credit_result, self.fraud_result, self.compliance_result]):
                raise ValueError("缺少必要的分析结果")
            
            # 准备决策分析输入
            summary = {
                "credit_analysis": self.credit_result,
                "fraud_analysis": self.fraud_result,
                "compliance_analysis": self.compliance_result
            }
            
            # 调用决策分析
            result = await self.decision_agent.make_loan_decision(json.dumps(summary, ensure_ascii=False))
            self.decision_result = result
            return result
            
        except Exception as e:
            self.logger.error(f"决策分析失败: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def run_final_decision(self, credit_result=None, fraud_result=None, compliance_result=None):
        """执行最终决策分析
        
        Args:
            credit_result: 信用分析结果
            fraud_result: 欺诈分析结果
            compliance_result: 合规分析结果
            
        Returns:
            Dict[str, Any]: 决策结果
        """
        session = None
        try:
            # 使用传入的结果或已存储的结果
            self.credit_result = credit_result or self.credit_result
            self.fraud_result = fraud_result or self.fraud_result
            self.compliance_result = compliance_result or self.compliance_result
            
            # 检查是否所有必要的分析都已完成
            if not all([self.credit_result, self.fraud_result, self.compliance_result]):
                raise ValueError("缺少必要的分析结果，请先完成所有分析")
            
            # 准备决策摘要
            summary = {
                "credit_analysis": self.credit_result,
                "fraud_analysis": self.fraud_result,
                "compliance_analysis": self.compliance_result
            }
            
            # 调用决策代理
            result = await self.decision_agent.make_loan_decision(json.dumps(summary, ensure_ascii=False))
            
            if isinstance(result, dict) and result.get('status') == 'success':
                self.logger.info("最终决策分析完成")
                return result
            else:
                error_msg = "最终决策分析失败: 无效的结果格式"
                self.logger.error(error_msg)
                return {"status": "error", "error": error_msg}
                
        except Exception as e:
            error_msg = f"最终决策分析错误: {str(e)}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg}
        finally:
            if session:
                await session.close()

    def display_decision_result(self, result):
        """显示决策结果"""
        if result is None:
            self.logger.error("未能获取决策结果")
            return
        
        try:
            # 如果result已经是字典，直接使用
            if isinstance(result, dict):
                result_dict = result
            else:
                # 如果是字符串，尝试解析为JSON
                result_dict = json.loads(result)
            
            if "assessment" in result_dict:
                print(result_dict["assessment"])
            else:
                print("\n=== 评估结果 ===")
                print(json.dumps(result_dict, ensure_ascii=False, indent=2))
        except (json.JSONDecodeError, TypeError, AttributeError):
            # 如果解析失败或不是预期格式，直接显示结果
            print("\n=== 评估结果 ===")
            print(result)
        
        self.logger.info("贷款评估流程已完成，结果已显示")

    def _configure_logging(self):
        """配置日志系统"""
        # 创建日志目录
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 生成日志文件名
        log_filename = os.path.join(log_dir, f"loan_application_{self.instance_id}.log")

        # 配置文件处理器
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # 配置logger
        self.logger = logging.getLogger(f"loan_application_{self.instance_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False

    async def get_evaluation_data(self) -> Dict[str, Any]:
        """获取评估所需的数据"""
        try:
            self.logger.info("开始获取评估数据")
            
            # 从收集的数据中提取评估所需的信息
            evaluation_data = {
                "申请人信息": {
                    "姓名": self.collected_data.get("name"),
                    "年龄": self.collected_data.get("age"),
                    "电话": self.collected_data.get("phone"),
                    "电子邮件": self.collected_data.get("email"),
                    "地址": self.collected_data.get("address")
                },
                "工作信息": {
                    "工作单位": self.collected_data.get("company"),
                    "职务": self.collected_data.get("position"),
                    "月收入": self.collected_data.get("monthly_income")
                },
                "贷款信息": {
                    "贷款金额": self.collected_data.get("loan_amount"),
                    "贷款用途": self.collected_data.get("loan_purpose"),
                    "贷款期限": self.collected_data.get("loan_term")
                },
                "证明文件": {
                    "工作证明": self.collected_data.get("employment_certificate"),
                    "银行流水": self.collected_data.get("bank_statement"),
                    "用户ID": self.collected_data.get("user_id")
                }
            }
            
            self.logger.info("评估数据准备完成")
            return evaluation_data
            
        except Exception as e:
            self.logger.error(f"获取评估数据失败: {str(e)}")
            raise

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口，确保资源被正确关闭"""
        try:
            # 关闭所有打开的会话和连接器
            if hasattr(self, '_sessions'):
                for session in self._sessions:
                    await session.close()
            if hasattr(self, '_connectors'):
                for connector in self._connectors:
                    await connector.close()
        except Exception as e:
            self.logger.error(f"关闭资源时发生错误: {str(e)}")

    def collect_expert_messages(self, messages):
        """收集专家的消息"""
        self.logger.info(f"开始收集专家消息，输入消息数量: {len(messages) if messages else 0}")
        
        expert_messages = []
        if not messages:
            self.logger.warning("输入的消息列表为空")
            return expert_messages
            
        for idx, msg in enumerate(messages):
            self.logger.info(f"处理第 {idx + 1} 条消息:")
            self.logger.info(f"  消息类型: {type(msg)}")
            
            # 检查消息对象的属性
            if hasattr(msg, 'name'):
                self.logger.info(f"  角色: {msg.name}")
            else:
                self.logger.warning(f"  消息对象没有name属性: {msg}")
                continue
                
            if hasattr(msg, 'content'):
                self.logger.info(f"  内容前50个字符: {msg.content[:50]}...")
            else:
                self.logger.warning(f"  消息对象没有content属性: {msg}")
                continue
            
            if msg.name in ['CreditExpert', 'FraudExpert', 'ComplianceExpert', 'DecisionAgent'] and msg.name != 'chat_manager':
                expert_messages.append({
                    'role': msg.name,
                    'content': msg.content
                })
                self.logger.info(f"  已添加 {msg.name} 的消息")
            else:
                self.logger.info(f"  跳过 {msg.name} 的消息")
        
        self.logger.info(f"消息收集完成，收集到 {len(expert_messages)} 条消息")
        return expert_messages

class DecisionAgent(autogen.AssistantAgent):
    def __init__(self, name, llm_config):
        super().__init__(
            name=name,
            system_message="""你是贷款决策专家，负责:
1. 执行初始分析
2. 协调其他专家的评估
3. 综合所有意见以及提供的决策结果做出最终决策，注意不要被信用分析结果，欺诈分析结果以及合规分析结果影响
4. 当流程完成时，在消息中包含'EVALUATION_COMPLETE'标记

评估流程：
1. 第一轮：请各专家分别提供初步评估意见
2. 第二轮：综合讨论并形成初步决策
3. 第三轮：发布最终决策并标注'EVALUATION_COMPLETE'

注意事项：
1. 每轮讨论都要确保所有专家都参与，需要等待所有专家都发表意见后才能进行总结
2. 引导讨论聚焦于关键风险点
3. 确保最终决策考虑了所有专家的意见
4. 只有在完成充分讨论后才能发布最终决策


EVALUATION_COMPLETE""",
            llm_config=llm_config
        )

class CreditExpert(autogen.AssistantAgent):
    def __init__(self, name, llm_config):
        super().__init__(
            name=name,
            system_message="""你是信用评估专家，负责:
1. 分析申请人的信用记录
2. 评估还款能力
3. 提供详细的信用评分报告

在你的发言中，你需要：
1. 直接使用提供的信用分析结果，不要被欺诈分析结果，合规分析结果以及决策结果影响
2. 首轮发言直接展示信用分析结果
3. 之后发言中结合信用分析结果自由讨论

注意点：
只负责信用评估相关的内容
不能代替其他专家发言
只能就信用相关问题进行讨论
""",
            llm_config=llm_config
        )

class FraudExpert(autogen.AssistantAgent):
    def __init__(self, name, llm_config):
        super().__init__(
            name=name,
            system_message="""你是反欺诈专家，负责:
1. 检查文件真实性
2. 识别可疑模式
3. 评估欺诈风险

在你的发言中，你需要：
1. 直接使用提供的欺诈分析结果，不要被信用分析结果，合规分析结果以及决策结果影响
2. 首轮发言直接展示欺诈分析结果
3. 之后发言中结合欺诈分析结果自由讨论

注意点：
只负责反欺诈相关的内容
不能代替其他专家发言
只能就反欺诈相关问题进行讨论

""",
            llm_config=llm_config
        )

class ComplianceExpert(autogen.AssistantAgent):
    def __init__(self, name, llm_config):
        super().__init__(
            name=name,
            system_message="""你是合规专家，负责:
1. 确保贷款申请符合监管要求
2. 审查所有必要文件
3. 验证合规性

在你的发言中，你需要：
1. 直接使用提供的合规分析结果，不要被信用分析结果，欺诈分析结果以及决策结果影响
2. 首轮发言直接展示合规分析结果
3. 之后发言中结合合规分析结果自由讨论

注意点：
只负责合规相关的内容
不能代替其他专家发言
只能就合规相关问题进行讨论

评估完成""",
            llm_config=llm_config
        )

# 注册函数
def register_functions(agent):
    """注册函数到代理"""
    # 创建LoanApplication实例
    loan_app = LoanApplication()
    
    # 定义函数映射
    function_map = {
        "run_initial_analysis": loan_app.run_initial_analysis,
        "run_credit_analysis": loan_app.run_credit_analysis,
        "run_fraud_analysis": loan_app.run_fraud_analysis,
        "run_final_decision": loan_app.run_final_decision
    }
    
    # 注册函数到执行
    agent.register_for_execution(function_map)
    
    # 设置llm_config中的functions
    agent.llm_config = {
        "config_list": config_list,
        "temperature": 0.7,
        "functions": [
            {
                "name": "run_initial_analysis",
                "description": "执行初始贷款分析",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "loan_data": {
                            "type": "string",
                            "description": "贷款申请数据"
                        }
                    },
                    "required": ["loan_data"]
                }
            },
            {
                "name": "run_credit_analysis",
                "description": "执行信用分析",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "certificate_file": {
                            "type": "string",
                            "description": "工作证明文件"
                        },
                        "bank_statement_file": {
                            "type": "string",
                            "description": "银行流水文件"
                        }
                    },
                    "required": ["certificate_file", "bank_statement_file"]
                }
            },
            {
                "name": "run_fraud_analysis",
                "description": "执行欺诈分析",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "用户ID"
                        }
                    },
                    "required": ["user_id"]
                }
            },
            {
                "name": "run_final_decision",
                "description": "执行最终决策",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "credit_result": {
                            "type": "string",
                            "description": "信用分析结果"
                        },
                        "fraud_result": {
                            "type": "string",
                            "description": "欺诈分析结果"
                        },
                        "compliance_result": {
                            "type": "string",
                            "description": "合规分析结果"
                        },
                        "decision_result": {
                            "type": "string",
                            "description": "决策结果"
                        }
                    }
                }
            }
        ]
    }

def is_expert_message(msg):
    expert_names = ['CreditExpert', 'FraudExpert', 'ComplianceExpert', 'DecisionAgent']
    return (
        hasattr(msg, 'name') and 
        msg.name in expert_names and
        msg.name != 'chat_manager'
    )

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='贷款申请系统')
    parser.add_argument('--test', action='store_true', help='运行测试模式')
    args = parser.parse_args()

    async with LoanApplication() as loan_app:
        if args.test:
            # 生成测试数据
            test_data = loan_app.generate_test_data()
            # 直接进入评估阶段
            await loan_app.phase_evaluation(test_data)
        else:
            # 正常运行流程
            await loan_app.run()

if __name__ == "__main__":
    asyncio.run(main()) 
