import os
import chardet
import datetime
import threading
from dotenv import load_dotenv
from my_tool import *
import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import Docx2txtLoader
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import SystemMessage


'''
这是一个学生的数据：{"study_hours_per_day": 4.6,"social_media_hours": 2,"netflix_hours": 3.6,"part_time_job": "Yes","attendance_percentage": 81.1,"sleep_hours": 6.8,"diet_quality": "Fair","exercise_frequency": 5,"parental_education_level": "Bachelor","internet_quality": "Average","mental_health_rating": 4,"extracurricular_participation": "No"}，跟据这个数据对他的成绩做预测，你再给出一些改进的建议
藜一般在几月播种？
藜怎么防治虫害？
描述一下藜麦的植株形状
'''
load_dotenv("API.env")

# 设置 OpenRouter API Key 和 Base URL
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# file_path = "limai.txt"
# with open(file_path, "rb") as f:
#     raw_data = f.read()
#     result = chardet.detect(raw_data)

# loader = Docx2txtLoader(r"C:\Users\23921\Documents\word/基于情感系数与常用词干预的个性对话生成.docx")
# docs = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=20)
# documents = text_splitter.split_documents(docs)

# model_name = r"C:\Users\23921\.cache\huggingface\hub\models--moka-ai--m3e-base\snapshots\764b537a0e50e5c7d64db883f2d2e051cbe3c64c"
# model_kwargs = {"device": "cuda"}
# encode_kwargs = {"normalize_text": True}
# search_kwargs = {"k": 10}
# embeddings = HuggingFaceEmbeddings(model_name=model_name,
#                                    model_kwargs=model_kwargs,
#                                    encode_kwargs=encode_kwargs,
#                                    )
# db = Chroma.from_documents(documents, embeddings)




# retriever = db.as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -- 初始化 LLM --
llm = ChatOpenAI(
    model="qwen-max",
    temperature=0.5,
    openai_api_base=openai.base_url,
    openai_api_key=openai.api_key,
    max_tokens=4096  # ✅ 限制生成的 token 数量
)

# 提醒事项存储
reminders = []

# 提醒工具函数
def add_reminder(reminder_info):
    """
    添加提醒事项
    格式："提醒内容,提醒时间"，例如："吃药,2026-03-03 12:00"
    """
    try:
        content, time_str = reminder_info.split(',', 1)
        reminder_time = datetime.datetime.strptime(time_str.strip(), "%Y-%m-%d %H:%M")
        reminders.append({"content": content.strip(), "time": reminder_time})
        # 启动一个线程来检查提醒
        def check_reminder():
            while True:
                now = datetime.datetime.now()
                print(f"当前系统时间：{now.strftime('%Y-%m-%d %H:%M:%S')}")  # 打印当前时间
                for reminder in reminders[:]:
                    if now >= reminder["time"]:
                        print(f"\n🔔 提醒：{reminder['content']}")
                        reminders.remove(reminder)
                import time
                time.sleep(60)  # 每分钟检查一次
        
        # 只启动一次检查线程
        if not hasattr(add_reminder, "thread_started"):
            thread = threading.Thread(target=check_reminder, daemon=True)
            thread.start()
            add_reminder.thread_started = True
        
        return f"已添加提醒：{content.strip()}，时间：{time_str.strip()}\n当前系统时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        return f"添加提醒失败：{str(e)}，请使用正确格式：提醒内容,提醒时间（例如：吃药,2026-03-03 12:00）"

# 查看所有提醒
def list_reminders():
    """
    查看所有已添加的提醒事项
    """
    if not reminders:
        return "当前没有提醒事项"
    result = "当前提醒事项：\n"
    for i, reminder in enumerate(reminders, 1):
        result += f"{i}. {reminder['content']} - {reminder['time'].strftime('%Y-%m-%d %H:%M')}\n"
    return result

# -- 封装工具 --
tools = [
    Tool(
        name="WeatherTool",
        func=get_weather,
        description="用于查询天气信息，输入城市名称（中文或英文）"
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="用于数学表达式的计算，例如：2 + 3 * (4 - 1)"
    ),
    Tool(
        name="TranslationTool",
        func=translate_to_chinese,
        description="翻译英文到中文，例如：What is artificial intelligence?"
    ),
    Tool(
        name="Studyadvicer",
        func=give_advices,
        description="跟据学生的学习习惯数据预测考试分数，并给每一个学生提出建议，预测分数来源于give_advices函数，建议则跟据生活习惯自己构思，语言生动委婉一点"
        ),
    Tool(
        name="DocumentQA",
        func=lambda q: qa({"question": q})["answer"],
        description="若遇到藜麦的问题，根据本地知识文档回答问题"
    ),
    Tool(
        name="ReminderTool",
        func=add_reminder,
        description="用于添加提醒事项，格式：提醒内容,提醒时间（例如：吃药,2026-03-03 12:00）。你需要将用户输入的自然语言时间转换为这种格式，例如将'今天下午4点'转换为'2026-03-03 16:00'"
    ),
    Tool(
        name="ListReminders",
        func=list_reminders,
        description="查看所有已添加的提醒事项"
    )
]

# qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)

# 打印当前系统时间
print(f"当前系统时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("提醒工具已就绪，您可以添加提醒事项了！")

# -- 初始化 Agent --
# 创建系统消息，包含当前时间信息
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
system_message = SystemMessage(content=f"你是一个智能助手，当前时间是 {current_time}。当用户要求添加提醒时，你需要将用户输入的自然语言时间转换为 'YYYY-MM-DD HH:MM' 格式，然后使用 ReminderTool 工具添加提醒。")

# 将系统消息添加到 memory 中
memory.chat_memory.messages.insert(0, system_message)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# -- 测试 Agent --
while  True:
    query = input("请输入你的问题：")
    if  query == "exit":
        break
    respond = agent.run({"input":query})
    print(respond)
