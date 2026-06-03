# 智能服装客服（LangChain RAG 升级版）

这是一个基于 LangChain 的服装购买与知识问答客服项目，已从基础 RAG 升级为"可检索 + 可推荐 + 可追溯"的电商客服 Agent。

## 项目简介

本项目面向服装电商场景，构建了一个智能客服系统，能够：

- **尺码推荐**：根据用户身高体重自动推荐合适尺码
- **商品推荐**：结合风格、季节、预算等条件推荐商品
- **售后问答**：提供退换货政策等标准化回答
- **知识库检索**：基于向量知识库回答洗护、颜色选择等常见问题

## 核心升级点

### 1. 混合检索

- 向量相似度检索 + MMR 多样性检索 + 关键词轻量重排
- 相比单一 `top-k`，能减少漏召回，提升回答稳定性

### 2. 业务工具层

- 内置 `BusinessTools`：
  - 尺码建议（身高/体重解析与规则推荐）
  - 商品推荐（风格/季节/预算的规则匹配）
  - 售后政策知识（退换货标准化回答）
- 业务工具结果与知识库检索结果融合后再交给大模型生成回答

### 3. 来源可追溯

- 检索内容会注入"可引用来源"，让客服回答更可解释，方便后续审计与质检

### 4. 会话级记忆

- 基于文件持久化的 `RunnableWithMessageHistory`，支持多 Session 历史隔离

### 5. 工程化包装

- `requirements.txt`：统一依赖
- `.env.example`：统一配置模板
- Streamlit UI 升级：会话 ID 管理、历史清空、示例问题

## 项目结构

```text
.
├── app_qa.py                # 客服问答前端（Streamlit）
├── app_file_uploader.py     # 知识库上传前端（Streamlit）
├── rag.py                   # RAG 主服务（混合检索 + 业务工具 + 会话记忆）
├── vector_stores.py         # 混合检索与重排（Similarity + MMR + 关键词重排）
├── business_tools.py        # 业务工具（尺码/搭配/售后）
├── product_catalog.py       # 内置商品样例库（6款商品）
├── knowledge_base.py        # 文本切片与向量写入（含 MD5 去重）
├── file_history_store.py    # 会话历史持久化（JSON 文件存储）
├── config_data.py           # 配置中心（支持环境变量覆盖）
├── .env.example             # 环境变量配置模板
├── data/                    # 知识库文本（尺码推荐/洗涤养护/颜色选择）
├── chroma_db/               # Chroma 向量数据库持久化目录
├── chat_history/            # 会话历史文件存储目录
└── requirements.txt         # Python 依赖
```

## 技术栈

| 类别 | 技术/库 | 用途 |
|------|---------|------|
| 前端框架 | Streamlit | 交互式用户界面 |
| 大语言模型 | DashScope 通义千问（qwen-plus） | 生成回答 |
| 嵌入模型 | DashScope Embeddings（text-embedding-v4） | 文本向量化 |
| 向量数据库 | Chroma | 存储和检索向量 |
| RAG 框架 | LangChain | RAG 流程编排与管理 |
| 会话记忆 | LangChain RunnableWithMessageHistory | 多会话历史隔离 |

## 混合检索实现详解（向量 Similarity + MMR + 关键词轻量重排）

本项目在 [`vector_stores.py`](vector_stores.py) 中实现了三阶段混合检索，替代传统"单一路径 top-k"。

### 1. 为什么单一 top-k 容易漏召回

单一向量 `similarity_search(query, k)` 常见问题：

- **语义相近但信息重复**：返回多段内容都在讲同一件事，覆盖面不够
- **业务关键词弱化**：电商场景里"退换货、尺码、预算"等词有时在纯语义里权重不够，容易丢关键片段
- **查询表达变化敏感**：用户换一种问法，top-k 的结果波动可能较大，导致回答稳定性下降

### 2. 本项目三阶段检索流程

输入问题 `query` 后按以下流程执行：

1. **向量相似度召回**：调用 `similarity_search(query, k=config.top_k)` 获取最语义相似的一批文档
2. **MMR 多样性召回**：调用 `max_marginal_relevance_search(query, k=config.mmr_k)` 获取"相关且彼此不那么重复"的文档
3. **合并去重 + 关键词轻量重排**：将两路结果合并，按 `page_content + metadata` 去重，再按关键词重叠分数排序，取前 `config.hybrid_top_k`

对应核心入口：

- 检索函数：`VectorStoreService.retrieve_documents(query)`
- 关键词打分函数：`VectorStoreService._keyword_overlap_score(query, text)`

### 3. MMR 在这里的作用

可以把 MMR 理解为"相关性 + 新信息量"的平衡：

- 只看相关性：容易拿到高度重复片段
- 加入多样性约束：更容易覆盖"尺码、颜色、洗护、售后"等不同子主题

这对客服场景非常关键，因为一轮回答通常需要同时包含多个维度的信息，而不是只命中一个点。

### 4. 关键词轻量重排是怎么做的

本项目没有引入额外 reranker 模型，而是做了轻量规则重排（低成本、易维护）：

- 将 `query` 和候选文档 `text` 做简单分词（按空白拆分）
- 计算重叠比例：`overlap = |query_tokens ∩ text_tokens| / |query_tokens|`
- 按重叠分数降序排序

**收益**：

- 可以补偿纯向量检索对高价值业务词的忽略
- 在不增加外部模型调用成本的情况下，提升命中稳定性

> 说明：中文场景下若要进一步提升效果，可替换为更好的分词策略（如 jieba）或接入专业 reranker。

### 5. 在 RAG 主链路中的接入方式

在 [`rag.py`](rag.py) 中：

1. `_retrieve_bundle(query)` 调用 `retrieve_documents(query)`
2. 将文档格式化为 `retrieved_context`（正文）和 `source_hints`（来源列表）
3. 与 `business_context`（业务工具输出）合并后注入 Prompt
4. 模型最终生成答案

这使得回答不只"语义相关"，还具备"业务可用性 + 来源可追溯"。

### 6. 可调参数与建议

配置位置：[`config_data.py`](config_data.py) 或 `.env`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TOP_K` | 4 | 相似度召回数量 |
| `MMR_K` | 4 | MMR 召回数量 |
| `HYBRID_TOP_K` | 5 | 最终输出给 LLM 的文档数 |
| `CHUNK_SIZE` | 800 | 文本切片大小 |
| `CHUNK_OVERLAP` | 120 | 切片重叠字符数 |

**建议**：

- 若回答经常"信息不全"：适当提高 `TOP_K` / `MMR_K`
- 若回答太啰嗦或噪声大：降低 `HYBRID_TOP_K`
- 若业务词命中弱：保留轻量重排并考虑升级为专用 reranker

## 业务工具说明

[`business_tools.py`](business_tools.py) 内置了三个业务工具，通过意图检测自动触发：

### 尺码建议

- 从用户输入中提取身高（cm）和体重（kg/斤）
- 根据规则映射到 S / M / L / XL / XXL 尺码
- 示例输入："我170cm/65kg，穿多大尺码？"

### 商品推荐

- 从用户输入中提取预算信息
- 基于风格、季节、颜色、价格等多维度打分排序
- 内置 6 款样例商品（针织开衫、卫衣、连衣裙、牛仔裤、风衣、毛衣）
- 示例输入："通勤风预算300元，有什么推荐？"

### 售后政策

- 提供标准化的退换货政策回答
- 覆盖 7 天无理由退换、质量问题包邮、定制商品规则
- 示例输入："这件衣服不合适可以退换吗？"

## 知识库管理

[`knowledge_base.py`](knowledge_base.py) 实现了知识库的文本切片与向量化写入：

- **MD5 去重**：通过 MD5 校验避免重复导入相同内容
- **智能切片**：使用 `RecursiveCharacterTextSplitter` 按自然段落分割
- **元数据记录**：记录来源文件名、创建时间、操作人

[`app_file_uploader.py`](app_file_uploader.py) 提供了 Streamlit 前端，支持上传 `.txt` / `.md` 文件并自动写入向量库。

## 快速开始

### 前置条件

- Python 3.9+
- 阿里云 DashScope API Key（[开通地址](https://help.aliyun.com/document_detail/2712195.html)）

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd longchain-main
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> 国内用户可使用镜像源加速：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 4. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，设置 `DASHSCOPE_API_KEY`。

### 5. 初始化知识库

将知识文档（如 `data/` 目录下的文本文件）通过知识库上传服务导入：

```bash
streamlit run app_file_uploader.py
```

### 6. 启动客服问答服务

```bash
streamlit run app_qa.py
```

## 使用示例

### 尺码咨询

> **用户**：我170cm/65kg，穿多大尺码？
>
> **客服**：尺码建议：根据你提供的 170cm / 65kg，建议优先试 L 码（不同版型可能上下浮动一码）。

### 商品推荐

> **用户**：通勤风预算300元，怎么选？
>
> **客服**：商品推荐：
> 1. 轻薄针织开衫（TOP-1001）| 价格: 239元 | 颜色: 米白/浅灰/雾蓝 | 风格: 通勤/简约
> 2. 高腰直筒牛仔裤（PANTS-3001）| 价格: 259元 | 颜色: 深蓝/水洗蓝 | 风格: 通勤/复古
> 搭配建议：上紧下松或同色系叠穿更容易显瘦和提升高级感。

### 售后咨询

> **用户**：这件衣服不合适可以退换吗？
>
> **客服**：售后政策参考：
> 1. 未洗涤未穿着且吊牌完整支持7天无理由退换
> 2. 质量问题支持包邮退换
> 3. 定制类商品不支持无理由退换

### 洗护咨询

> **用户**：针织毛衣怎么洗？会起球吗？
>
> **客服**：基于知识库回答洗涤养护相关问题。

## 配置参考

所有配置项均支持通过环境变量覆盖，详见 [`.env.example`](.env.example)：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `DASHSCOPE_API_KEY` | - | 阿里云 DashScope API 密钥 |
| `COLLECTION_NAME` | rag | Chroma 集合名称 |
| `PERSIST_DIRECTORY` | ./chroma_db | 向量数据库持久化路径 |
| `CHUNK_SIZE` | 800 | 文本切片大小 |
| `CHUNK_OVERLAP` | 120 | 切片重叠字符数 |
| `TOP_K` | 4 | 相似度检索数量 |
| `MMR_K` | 4 | MMR 检索数量 |
| `HYBRID_TOP_K` | 5 | 最终输出文档数 |
| `EMBEDDING_MODEL_NAME` | text-embedding-v4 | 嵌入模型 |
| `CHAT_MODEL_NAME` | qwen-plus | 对话模型 |
| `DEFAULT_SESSION_ID` | user_001 | 默认会话 ID |

## 开发计划

- [ ] 加入商品库存/价格实时接口（Tools 调用 ERP 或商城 API）
- [ ] 引入 reranker 模型（如 bge-reranker）替换当前轻量重排
- [ ] 增加用户画像记忆（尺码偏好、颜色偏好、预算区间）做个性化推荐
- [ ] 接入埋点与评估集（回答准确率、转化率、平均响应时长）形成闭环优化
- [ ] 支持多模态（商品图片识别与展示）
- [ ] 增加更多业务工具（如物流查询、优惠券查询）
