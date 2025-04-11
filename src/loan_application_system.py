import os
import autogen
from dotenv import load_dotenv
import openai
import asyncio  # 添加asyncio模块
from credit_agent import CreditAnalysisAgent  # 导入CreditAnalysisAgent
from fraud_agent import FoundryIncomeAgent  # 导入fraud_analysis_workflow
from decision_agent import LoanDecisionAgent  # 导入LoanDecisionAgent

# Load environment variables
load_dotenv()

# Ensure OpenAI library uses the correct configuration
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Configure models for autogen
config_list = [
    {
        "model": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "api_type": "azure",
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
}

class LoanApplication:
    def __init__(self):
        self.collected_data = {}
        self.evaluation_results = {}
 
    def run(self):
        print("=== 索尔维亚首都银行贷款评估系统 ===")
        print("系统启动中...")
        
        # 第一阶段: 信息收集
        print("\n第一阶段: 信息收集")
        success = self.phase_data_collection()
        
        # 如果数据收集成功，进入评估阶段
        if success and self.collected_data:
            print("\n第二阶段: 贷款评估")
            self.phase_evaluation()
            
            # 展示最终结果
            if self.evaluation_results and "未能获取" not in self.evaluation_results and "评估失败" not in self.evaluation_results:
                print("\n=== 贷款申请评估流程已完成 ===")
                print("感谢您使用索尔维亚首都银行贷款评估系统!")
            else:
                print("\n=== 评估过程未能正常完成 ===")
                print("请联系客服获取帮助。")
        else:
            print("\n数据收集未成功完成，无法进行评估。")
            print("请重新启动系统并完成信息收集。")
            
        print("\n系统即将退出...")
        
    def phase_data_collection(self):
        # 创建数据收集agent
        data_collector = autogen.AssistantAgent(
            name="DataCollector",
            system_message="""你是索尔维亚首都银行的贷款审核助手。
            首先，主动欢迎用户并介绍自己，表明你是索尔维亚首都银行的贷款服务助手，询问用户有什么可以帮助的。
            
            你需要有序地收集以下信息:
            1. 姓名
            2. 年龄
            3. 电话
            4. 电子邮件
            5. 地址
            6. 工作单位
            7. 职务
            8. 月收入
            9. 贷款金额
            10. 贷款用途
            11. 贷款期限（月）
            12. 房屋地址
            13. 贷款开始日期
            14. 房屋面积（平方米）
            
            请一个一个地提问，保持对话流畅友好。在完成所有必要信息的收集之前，继续对话。
            分析提供数据是否符合索尔维亚首都银行贷款申请要求，如果符合，则继续收集信息，否则，请用户修改信息。
            如果用户一次性提交了所有信息，直接分析数据是否符合索尔维亚首都银行贷款申请要求，如果符合，则继续收集信息，否则，请用户修改信息。

            当所有信息收集完毕，请提供格式化摘要:
            收集的数据摘要:
            - 姓名: [姓名]
            - 年龄: [年龄]
            - 电话: [电话]
            - 电子邮件: [电子邮件]
            - 地址: [地址]
            - 工作单位: [工作单位]
            - 职务: [职务]
            - 月收入: [月收入]
            - 贷款金额: [贷款金额]
            - 贷款用途: [贷款用途]
            - 贷款期限: [贷款期限]
            - 房屋地址: [房屋地址]
            - 贷款开始日期: [贷款开始日期]
            - 房屋面积: [房屋面积]
            
            收集完所有信息请添加标记"DATA_COLLECTION_COMPLETE"表示数据收集已完成。此句话不用告知用户
            DATA_COLLECTION_COMPLETE单独一行，不要与任何其他内容混杂在一起。
            """,
            llm_config=llm_config,
        )
        
        # 创建用户代理，接收用户输入
        user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="ALWAYS",  # 一直请求人工输入
            is_termination_msg=lambda x: "DATA_COLLECTION_COMPLETE" in x.get("content", ""),
            code_execution_config=False
        )
        
        # 提示用户输入信息
        print("\n=== 索尔维亚首都银行贷款申请系统 ===")
        print("欢迎使用索尔维亚首都银行贷款申请系统。")
        print("我将帮助您收集贷款申请所需的信息。请根据提示完成信息输入。\n")
        
        try:
            # 启动与数据收集Agent的对话
            print("正在连接贷款顾问，请稍候...\n")
            
            # 这里添加初始消息，让DataCollector开始对话
            chat_result = user_proxy.initiate_chat(
                data_collector,
                message="我想申请贷款"
            )
            
            # 提取收集到的数据
            final_message = None
            for message in reversed(user_proxy.chat_messages[data_collector]):
                if "DATA_COLLECTION_COMPLETE" in message.get("content", ""):
                    final_message = message.get("content", "")
                    break
            
            if final_message:
                # 从最终消息中提取结构化数据
                self.collected_data = self.parse_collected_data(final_message)
                
                # 确认收集到的数据
                print("\n收集的数据摘要:")
                loan_application = LoanApplication()
                for key, value in loan_application.collected_data.items():
                    print(f"- {key}: {value}")
                
                # 询问用户是否同意
                agreement = input("\n索尔维亚首都银行将使用以上个人信息进行贷款审核，您是否同意? (同意/不同意): ")
                
                if agreement == "同意":
                    print("\n信息收集完成!")
                    print("正在进入评估阶段...")
                    return True
                else:
                    print("\n您未同意信息使用条款，申请流程终止。")
                    return False
            else:
                print("\n未能成功完成数据收集。")
                return False
                
        except Exception as e:
            print(f"数据收集过程中出错: {e}")
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
    
    def phase_evaluation(self):
        # 打印评估开始提示
        print("\n=== 开始贷款评估阶段 ===")
        print("系统正在启动贷款评审团队，这可能需要一些时间，请耐心等待...\n")
        
        # 准备评估数据
        loan_data = ""
        for key, value in self.collected_data.items():
            loan_data += f"- {key}: {value}\n"
        
        print(f'loan_data:{loan_data}')

        # 创建信用分析后端实例
        credit_analysis_instance = CreditAnalysisAgent()
        
        # 封装异步调用函数
        def run_credit_analysis(certificate_file, bank_statement_file):
            """同步包装器，调用异步信用分析功能"""
            certificate_file = "yinhangliushui.png"
            bank_statement_file = "zhangsanzaizhi.png"
            result = asyncio.run(credit_analysis_instance.analyze_credit(
                certificate_file, bank_statement_file))
            return f"信用分析结果: {result['origin_result']['assessment']}" if result['origin_result']['status'] == "success" else f"信用分析失败: {result['origin_result'].get('message', '未知错误')}"


        fraud_agent = FoundryIncomeAgent()
        # 封装欺诈分析异步调用函数
        def run_fraud_analysis(user_id):
            """同步包装器，调用异步欺诈分析功能"""
            # 使用固定的测试ID
            test_user_id = "08c6c7e1-023d-4d21-a3d2-bb2d32efc7ca"
            print(f"正在进行欺诈分析，用户ID: {test_user_id}")
            result = asyncio.run(fraud_agent.fraud_analysis_workflow(test_user_id))
            if isinstance(result, str):
                return f"欺诈分析结果: {result}"
            elif isinstance(result, dict) and 'status' in result and result['status'] == 'error':
                return f"欺诈分析失败: {result.get('message', '未知错误')}"
            else:
                return f"欺诈分析结果: {result}"
                
        # 创建决策分析后端实例
        decision_analysis_instance = LoanDecisionAgent()
        
        # 创建专家结果跟踪字典
        self.expert_results = {
            "credit": None,
            "fraud": None,
            "compliance": None
        }
        
        # 封装初始决策分析异步调用函数
        def run_initial_analysis(loan_data):
            """同步包装器，调用决策agent进行初始分析"""
            # 在第一轮，仅需提供申请基本信息
            summary = f"贷款申请初步分析:\n{loan_data}"
            print("正在进行初始决策分析...")
            result = asyncio.run(decision_analysis_instance.make_loan_decision(summary))
            if isinstance(result, dict) and result.get('status') == 'success':
                return f"初始分析结果: {result.get('decision_result', '无结果')}"
            else:
                return f"初始分析失败: {str(result)}"
        
        # 封装最终决策异步调用函数
        def run_final_decision(credit_result=None, fraud_result=None, compliance_result=None):
            """同步包装器，调用决策agent进行最终决策"""
            # 汇总所有专家的分析结果
            summary = "贷款申请综合评估:\n"
            
            # 使用传入的参数或已存储的结果
            credit = credit_result or self.expert_results.get("credit", "未提供信用评估")
            fraud = fraud_result or self.expert_results.get("fraud", "未提供欺诈评估") 
            compliance = compliance_result or self.expert_results.get("compliance", "未提供合规评估")
            
            summary += f"信用评估: {credit}\n"
            summary += f"欺诈评估: {fraud}\n"
            summary += f"合规评估: {compliance}\n"
            
            print("正在进行最终决策分析...")
            result = asyncio.run(decision_analysis_instance.make_loan_decision(summary))
            if isinstance(result, dict) and result.get('status') == 'success':
                return f"最终决策: {result.get('decision_result', '无结果')}"
            else:
                return f"最终决策失败: {str(result)}"
            
        # 创建评估agents
        
        # 主决策agent - 作为团队协调者
        decision_agent = autogen.AssistantAgent(
            name="DecisionAgent",
            system_message="""你是索尔维亚首都银行的贷款评估专家，也是贷款审批团队的协调者。
            
            发言顺序严格控制:
            1. 第1轮对话：你首先发言，介绍申请案例并分配任务
            2. 之后保持沉默，直到第5轮对话
            3. 第5轮对话：你最后发言，综合各专家意见提供最终建议
            
            在第1轮的发言中，你需要：
            1. 使用run_initial_analysis(loan_data)函数进行初步分析，但在你的回复中，不要提及你使用了什么函数进行分析。
            2. 简要介绍申请案例
            3. 分配任务给各专家
            4. 明确表示由信用专家(CreditExpert)进行下一轮评估
            
            在第5轮的发言中，请使用run_final_decision()函数汇总各专家意见，并按以下格式提供你的最终建议:
            
            最终建议: [同意/建议修改/需要更多信息]
            理由: [基于所有专家意见的分析]
            贷款条件: [如适用]
            
            最后请添加标记"EVALUATION_COMPLETE"，这是系统识别评估完成的信号。
            """,
            llm_config=llm_config,
        )
        
        # 向decision_agent注册函数
        decision_agent.register_function(
            function_map={
                "run_initial_analysis": run_initial_analysis,
                "run_final_decision": run_final_decision
            }
        )
        
        # 信用评估agent
        credit_agent = autogen.AssistantAgent(
            name="CreditExpert",
            system_message="""你是索尔维亚首都银行的信用评估专家。
            
            发言顺序严格控制:
            1. 第2轮对话：你进行发言，提供信用评估
            2. 其他轮次保持沉默
            
            在你的发言中：
            1. 评估申请人的信用状况和还款能力
            2. 给出风险评级和建议
            3. 明确表示由欺诈专家(FraudExpert)进行下一轮评估
            
            评估格式:
            信用评估:
            风险等级: [低/中/高]
            说明: [你的评估分析]
            
            你要使用run_credit_analysis(certificate_file, bank_statement_file)函数
            来获取深度的信用分析结果。该函数会返回包含信用等级(A/B/C)和风险摘要的分析报告。
            
            示例调用:
            run_credit_analysis("ZhangSanZaiZhi.png", "ZhangSanLiuShui.png")
            
            在分析贷款申请时，你应该:
            1. 首先调用run_credit_analysis函数进行深度分析
            2. 将分析结果整合到你的评估中
            3. 提供最终的信用评估结论
            但在你的回复中，不要提及你使用了什么函数进行分析，直接分享分析结果即可。
            """,
            llm_config=llm_config,
        )
        
        # 向credit_agent注册函数
        credit_agent.register_function(
            function_map={
                "run_credit_analysis": run_credit_analysis
            }
        )
        
        # 欺诈检测agent
        fraud_agent = autogen.AssistantAgent(
            name="FraudExpert",
            system_message="""你是索尔维亚首都银行的欺诈风险评估专家。
            
            发言顺序严格控制:
            1. 第3轮对话：你进行发言，提供欺诈风险评估
            2. 其他轮次保持沉默
            
            在你的发言中：
            1. 分析潜在的欺诈风险
            2. 评估申请材料的真实性
            3. 明确表示由合规专家(ComplianceExpert)进行下一轮评估
            
            评估格式:
            欺诈风险评估:
            风险等级: [低/中/高]
            说明: [你的评估分析]
            
            你要使用run_fraud_analysis(user_id)函数获取深度欺诈分析结果。
            该函数会返回包含风险分析和欺诈风险摘要的分析报告。
            
            示例调用:
            run_fraud_analysis("08c6c7e1-023d-4d21-a3d2-bb2d32efc7ca")
            
            在评估申请时，你应该:
            1. 调用run_fraud_analysis函数进行深度分析
            2. 将分析结果整合到自己的评估中
            3. 提供最终的欺诈风险评估结论
            """,
            llm_config=llm_config,
        )
        
        # 向fraud_agent注册函数
        fraud_agent.register_function(
            function_map={
                "run_fraud_analysis": run_fraud_analysis
            }
        )
        
        # 合规审查agent
        compliance_agent = autogen.AssistantAgent(
            name="ComplianceExpert",
            system_message="""你是索尔维亚首都银行的合规审查专家。
            
            发言顺序严格控制:
            1. 第4轮对话：你进行发言，提供合规审查
            2. 其他轮次保持沉默
            
            在你的发言中：
            1. 确保申请符合监管要求
            2. 审查文件完整性与合规性
            3. 明确表示由决策专家(DecisionAgent)做出最终决策
            
            评估格式:
            合规审查:
            合规状态: [符合/不符合/需补充]
            说明: [你的评估分析]
            """,
            llm_config=llm_config,
        )
        
        # 创建贷款评审团队
        loan_review_team = autogen.GroupChat(
            agents=[decision_agent, credit_agent, fraud_agent, compliance_agent],
            messages=[],
            max_round=6  # 精确控制6轮对话：1.决策专家 2.信用专家 3.欺诈专家 4.合规专家 5.决策专家总结
        )
        
        # 设置决策专家为第一个发言者
        loan_review_team.next_agent = decision_agent
        
        loan_review_manager = autogen.GroupChatManager(groupchat=loan_review_team, llm_config=llm_config)
        
        # 准备评估数据
        # 申请人提供的雇佣证明文件: {self.collected_data.get('employment_certificate', 'ZhangSanZaiZhi.png')}
        # 申请人提供的银行流水文件: {self.collected_data.get('bank_statement', 'ZhangSanLiuShui.png')}
        # 申请人用户ID: {self.collected_data.get('user_id', '08c6c7e1-023d-4d21-a3d2-bb2d32efc7ca')}
        loan_request = f"""
        申请数据:
        {loan_data}
        
        
        
        请对这份贷款申请进行评估。请严格按照以下轮次发言:
        第1轮：决策专家(DecisionAgent)介绍并分配任务
        第2轮：信用专家(CreditExpert)进行信用评估
        第3轮：欺诈专家(FraudExpert)进行欺诈风险评估
        第4轮：合规专家(ComplianceExpert)进行合规审查
        第5轮：决策专家(DecisionAgent)提供最终决策
        
        每位专家只在自己的指定轮次发言，其他时间保持沉默。
        请开始评估流程。
        """
        
        # 启动团队评审流程
        print("正在提交贷款申请到评审团队...")
        
        try:
            # 创建一个不依赖termination的用户代理，完全依靠max_round来控制对话结束
            basic_user_proxy = autogen.UserProxyAgent(
                name="BasicUserProxy",
                human_input_mode="NEVER",  # 确保不需要人工干预
                is_termination_msg=None,   # 不使用终止条件
                llm_config=None,           # 不需要LLM配置
                system_message="仅用于启动对话，不参与评估。"
            )
            
            # 启动团队评审流程
            print("启动评估流程，最多进行6轮对话...")
            basic_user_proxy.initiate_chat(
                loan_review_manager,
                message=loan_request
            )
            print("评估对话已完成！")
            
            # 提取评估结果
            messages = loan_review_team.messages
            print(f"处理{len(messages)}条消息...")
            
            final_decision = self.extract_decision_from_messages(messages)
            
            # 保存最终评估结果
            self.evaluation_results = final_decision
            
            print("\n=== 评估完成! ===")
            print("最终建议:\n", self.evaluation_results)
            
        except Exception as e:
            print(f"评估过程中出错: {e}")
            print("无法完成贷款评估，请联系系统管理员。")
            self.evaluation_results = "评估失败: " + str(e)
    
    def extract_decision_from_messages(self, messages):
        """从对话历史中提取决策结果"""
        print(f"\n开始提取决策结果，处理{len(messages)}条消息...")
        
        # 打印消息内容以便调试
        # for i, msg in enumerate(messages):
        #     if isinstance(msg, dict):
        #         name = msg.get("name", "未知")
        #         content_length = len(str(msg.get("content", "")))
        #         print(f"消息 {i+1} - 发送者: {name}, 内容长度: {content_length}")
                
        #         # 捕获专家评估结果
        #         if name == "CreditExpert" and self.expert_results["credit"] is None:
        #             self.expert_results["credit"] = msg.get("content", "")
        #             print(f"已捕获信用专家评估，长度: {len(self.expert_results['credit'])}")
        #         elif name == "FraudExpert" and self.expert_results["fraud"] is None:
        #             self.expert_results["fraud"] = msg.get("content", "")
        #             print(f"已捕获欺诈专家评估，长度: {len(self.expert_results['fraud'])}")
        #         elif name == "ComplianceExpert" and self.expert_results["compliance"] is None:
        #             self.expert_results["compliance"] = msg.get("content", "")
        #             print(f"已捕获合规专家评估，长度: {len(self.expert_results['compliance'])}")
        #     else:
        #         print(f"消息 {i+1} - 类型: {type(msg)}")
        
        # 寻找最后一条决策专家的消息
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("name") == "DecisionAgent":
                content = message.get("content", "")
                if content and isinstance(content, str):
                    print(f"找到决策专家的最终消息")
                    return content
        
        # 如果没有找到决策专家的消息，尝试查找任何包含"最终建议"的消息
        for message in reversed(messages):
            if isinstance(message, dict):
                content = message.get("content", "")
                if content and isinstance(content, str) and "最终建议:" in content:
                    print(f"找到包含最终建议的消息")
                    return content
        
        # 尝试从最后一条消息提取结果
        if messages and isinstance(messages[-1], dict) and "content" in messages[-1]:
            return messages[-1]["content"]
        
        return "未能获取完整评估结果"
    
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
            "employment_certificate": "ZhangSanZaiZhi.png",
            "bank_statement": "ZhangSanLiuShui.png",
            "user_id": "08c6c7e1-023d-4d21-a3d2-bb2d32efc7ca"
        }
        print("已生成测试数据:")
        loan_application = LoanApplication()
        for key, value in loan_application.collected_data.items():
            print(f"- {key}: {value}")
        return loan_application.collected_data


if __name__ == "__main__":
    # 检查是否有测试命令行参数
    import sys
    loan_app = LoanApplication()
    
    # 如果有--test参数，则生成测试数据并直接进入评估阶段
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("=== 索尔维亚首都银行贷款评估系统 (测试模式) ===")
        print("\n生成测试数据中...")
        loan_app.generate_test_data()
        
        print("\n启动评估阶段...")
        loan_app.phase_evaluation()
        
        # 展示最终结果
        if loan_app.evaluation_results and "未能获取" not in loan_app.evaluation_results and "评估失败" not in loan_app.evaluation_results:
            print("\n=== 贷款申请评估流程已完成 ===")
            print("感谢您使用索尔维亚首都银行贷款评估系统!")
        else:
            print("\n=== 评估过程未能正常完成 ===")
            print("请联系客服获取帮助。")
    else:
        # 正常运行
        loan_app.run() 