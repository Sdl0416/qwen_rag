# Qwen + RAG 企业技术文档知识库问答系统

一个基于 **Qwen + Embedding + FAISS** 实现的本地技术文档知识库问答项目。

系统支持读取 PDF、TXT、Markdown 文档，对文本进行切分并调用 Embedding 模型生成向量，使用 FAISS 建立向量索引；用户提出问题后，系统通过语义相似度检索召回相关文档片段，并将检索结果作为上下文交给 Qwen 大语言模型生成答案，从而实现基于企业/技术文档的检索增强生成（RAG）问答。

项目还提供了 **直接调用 Qwen** 与 **Qwen + RAG** 的对比功能，便于观察引入本地知识库前后的回答差异。

---

## 功能特性

- 支持 PDF / TXT / Markdown 文档读取
- PDF 按页提取文本并保留页码信息
- 文本清洗与 Chunk 分块
- 支持 Chunk Overlap，减少上下文被截断的问题
- 调用 Qwen Embedding 模型生成文本向量
- 使用 FAISS 构建本地向量索引
- 使用向量归一化 + Inner Product 实现余弦相似度检索
- 支持 Top-K 相关文档片段召回
- 使用 Qwen 大语言模型生成基于知识库的回答
- 回答中可显示文档来源、页码和相似度
- 支持单次问答与命令行连续问答
- 支持“直接 Qwen”与“Qwen + RAG”效果对比
- 支持通过 `.env` 管理 API Key、模型名称与接口地址

---

## 技术栈

- Python
- Qwen
- OpenAI Compatible API
- Qwen Embedding
- FAISS
- NumPy
- PyPDF
- python-dotenv

---

## RAG 工作流程

```text
PDF / TXT / MD 文档
        │
        ▼
   文档解析与文本提取
        │
        ▼
   文本清洗与 Chunk 切分
        │
        ▼
     Embedding 模型
        │
        ▼
  文本向量 + 元数据
        │
        ▼
      FAISS 索引
        │
        └─────────────────────┐
                              │
用户问题                     │
   │                          │
   ▼                          │
Embedding                     │
   │                          │
   ▼                          │
Top-K 相似度检索 ◄────────────┘
   │
   ▼
召回相关文档片段
   │
   ▼
构造带知识库上下文的 Prompt
   │
   ▼
Qwen 大语言模型
   │
   ▼
答案 + 文档来源 / 页码
```

---

## 项目结构

```text
.
├── rag_app.py                 # RAG 核心代码及命令行入口
├── 01_build_index.py          # 构建知识库索引示例
├── 02_chat.py                 # 单次 RAG 问答示例
├── 03_chat_loop.py            # 连续交互式问答
├── 04_compare.py              # 直接 Qwen 与 RAG 对比
├── test_qwen.py               # Qwen 对话接口测试
├── test_embedding.py          # Embedding 接口测试
├── requirements.txt           # Python 依赖
├── .env.example               # 环境变量配置模板
├── .gitignore
├── docs/                      # 知识库原始文档
│   ├── 01_智能摄像头产品说明书.pdf
│   ├── 02_NVR录像存储系统操作手册.pdf
│   ├── 03_门禁一体机配置指南.pdf
│   ├── 04_企业AI知识库使用手册.pdf
│   ├── 05_售后故障排查FAQ.pdf
│   ├── 06_账号权限与信息安全规范.pdf
│   └── 07_RAG测试问题与标准答案.txt
└── index/                     # 构建后的 FAISS 索引及元数据
    ├── faiss.index
    └── metadata.json
```

> `index/` 和 `.env` 已在 `.gitignore` 中忽略。上传到 Git 仓库时，不建议提交 API Key 或其他敏感配置。

---

## 环境安装

建议使用 Python 虚拟环境。

### 1. 创建虚拟环境

```bash
python -m venv .venv
```

Linux / macOS：

```bash
source .venv/bin/activate
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

项目主要依赖：

```text
openai>=1.52
faiss-cpu>=1.8
numpy>=1.26
pypdf>=5.0
python-dotenv>=1.0
```

---

## 配置环境变量

复制配置模板：

Linux / macOS：

```bash
cp .env.example .env
```

Windows：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`：

```env
DASHSCOPE_API_KEY=你的_API_Key

DASHSCOPE_BASE_URL=你的_Model_Studio_OpenAI_Compatible_Base_URL

QWEN_CHAT_MODEL=qwen-plus
QWEN_EMBED_MODEL=qwen3.7-text-embedding
QWEN_EMBED_DIM=1024
```

其中：

- `DASHSCOPE_API_KEY`：Model Studio API Key
- `DASHSCOPE_BASE_URL`：OpenAI Compatible API 地址
- `QWEN_CHAT_MODEL`：用于答案生成的 Qwen 模型
- `QWEN_EMBED_MODEL`：用于文本向量化的 Embedding 模型
- `QWEN_EMBED_DIM`：Embedding 向量维度

> 不要将真实 `.env` 文件或 API Key 提交到公开仓库。

---

## 快速开始

### 1. 准备知识库文档

将需要检索的文档放入：

```text
docs/
```

当前代码支持：

```text
.pdf
.txt
.md
```

PDF 会按页读取，因此检索结果可以保留来源页码。

---

### 2. 构建知识库索引

可以直接使用：

```bash
python 01_build_index.py
```

该脚本当前设置：

```text
chunk_size = 300
overlap = 50
```

也可以通过主程序自定义：

```bash
python rag_app.py build \
  --docs docs \
  --chunk-size 300 \
  --overlap 50
```

构建流程包括：

1. 扫描 `docs/`
2. 解析 PDF / TXT / MD
3. 文本清洗
4. Chunk 切分
5. 调用 Embedding 模型
6. 对向量进行 L2 Normalize
7. 写入 FAISS `IndexFlatIP`
8. 保存索引和 Chunk 元数据

完成后会生成：

```text
index/faiss.index
index/metadata.json
```

当前示例索引使用：

```text
Embedding 模型：qwen3.7-text-embedding
向量维度：1024
Chunk Size：300
Overlap：50
Chunk 数量：22
```

---

## 单次 RAG 问答

运行示例：

```bash
python 02_chat.py
```

示例问题：

```text
QM-Cam X3最高分辨率是多少？
```

程序会输出：

- 用户问题
- Qwen + RAG 回答
- Top-K 检索来源
- 文档名称
- PDF 页码
- 相似度
- 召回文本内容

也可以直接使用主程序：

```bash
python rag_app.py ask "QM-Cam X3最高分辨率是多少？"
```

修改召回数量：

```bash
python rag_app.py ask \
  "QM-Cam X3最高分辨率是多少？" \
  --top-k 5
```

---

## 连续交互问答

运行：

```bash
python 03_chat_loop.py
```

或者：

```bash
python rag_app.py chat --top-k 3
```

进入命令行问答：

```text
===== Qwen + RAG 知识库问答 =====
输入 exit / quit 退出。

问题>
```

退出方式：

```text
exit
quit
q
```

---

## 对比直接 Qwen 与 RAG

项目提供了对比脚本：

```bash
python 04_compare.py
```

也可以使用：

```bash
python rag_app.py compare \
  "QM-Cam X3在没有DHCP服务时使用什么IP？"
```

程序会分别输出：

```text
直接 Qwen
```

以及：

```text
Qwen + RAG
```

这样可以直观看到，在涉及企业内部资料、产品参数、配置说明等问题时，引入知识库检索后模型回答的差异。

---

## 文档解析

### PDF

项目使用：

```python
pypdf.PdfReader
```

逐页提取 PDF 文本，并保存：

```text
source
page
text
```

因此检索结果能够显示原始文件名称和页码。

### TXT / Markdown

TXT 和 Markdown 文档使用 UTF-8 方式读取，并记录文件来源。

---

## Chunk 切分

当前项目采用基于字符长度的滑动窗口切分方式。

核心参数：

```text
chunk_size
overlap
```

例如：

```text
chunk_size = 300
overlap = 50
```

相邻 Chunk 保留一部分重叠内容，可以降低关键语义刚好位于切分边界时的信息损失。

当前实现较轻量，后续可以继续升级为：

- 按标题切分
- 按段落切分
- 按句子切分
- Token-aware Chunking
- Recursive Chunking

---

## Embedding 与 FAISS

文本经过 Embedding 模型后转换为向量：

```text
Text → Embedding Vector
```

项目当前使用：

```text
qwen3.7-text-embedding
```

默认维度：

```text
1024
```

随后执行 L2 Normalize：

```python
faiss.normalize_L2(vectors)
```

并使用：

```python
faiss.IndexFlatIP
```

建立 FAISS 索引。

由于向量经过归一化，因此 Inner Product 可以作为余弦相似度进行 Top-K 检索。

---

## 查询流程

当用户输入问题时，系统执行：

```text
用户问题
  ↓
问题 Embedding
  ↓
FAISS Top-K 检索
  ↓
相关 Chunk
  ↓
构建 RAG Prompt
  ↓
Qwen
  ↓
最终回答
```

检索结果包含：

```python
{
    "source": "文档名称",
    "page": 1,
    "text": "召回文本",
    "score": 0.82
}
```

---

## Prompt 约束

项目在 RAG Prompt 中要求模型：

1. 严格依据知识库资料回答
2. 不使用资料之外的事实补全答案
3. 如果资料不足，明确说明知识库信息不足
4. 回答尽量简洁、条理清晰
5. 在答案末尾标记使用的资料编号

这种方式能够降低知识库场景中的无依据生成问题。

---

## 主程序命令说明

### 构建索引

```bash
python rag_app.py build
```

参数：

| 参数           | 默认值 | 说明                |
| -------------- | -----: | ------------------- |
| `--docs`       | `docs` | 知识库文档目录      |
| `--chunk-size` |  `700` | Chunk 字符长度      |
| `--overlap`    |  `120` | 相邻 Chunk 重叠长度 |

例如：

```bash
python rag_app.py build \
  --docs docs \
  --chunk-size 300 \
  --overlap 50
```

### 交互问答

```bash
python rag_app.py chat --top-k 3
```

### 单次问答

```bash
python rag_app.py ask "你的问题" --top-k 3
```

### 直接 Qwen / RAG 对比

```bash
python rag_app.py compare "你的问题" --top-k 3
```

---

## 示例

用户提问：

```text
QM-Cam X3在没有DHCP服务时使用什么IP？
```

RAG 系统首先从本地技术文档中检索相关内容，再将召回结果发送给 Qwen。

最终输出除了答案外，还会显示类似：

```text
===== 检索来源 =====
[1] 01_智能摄像头产品说明书.pdf 第1页 score=...
    ...
```

相比直接调用大语言模型，这种方式可以让回答更多地依据指定的企业内部资料，而不是完全依赖模型自身已有知识。

---

## 测试脚本

### 测试 Qwen 对话接口

```bash
python test_qwen.py
```

用于验证：

- API Key 是否有效
- Base URL 是否正确
- Qwen Chat 模型是否可以正常调用

### 测试 Embedding

```bash
python test_embedding.py
```

用于验证：

- Embedding 模型是否可正常调用
- 返回向量维度是否正确

---

## 安全说明

项目使用 `.env` 保存 API Key。

`.gitignore` 中已经包含：

```text
.env
.venv/
__pycache__/
index/
```

建议：

- 不要将真实 API Key 写入代码
- 不要将 `.env` 上传到 GitHub
- 如果 API Key 曾经被公开，应及时重新生成
- 公开项目时只保留 `.env.example`

---

## 可扩展方向

当前版本是一个轻量级 RAG 原型，可以继续增加：

- Web 可视化问答界面
- Streamlit / Gradio 前端
- 多轮对话与历史消息管理
- 文档上传功能
- Word / Excel / PPT 等更多文档格式
- OCR 扫描 PDF 支持
- Token 级智能文本切分
- Metadata Filter
- Hybrid Search
- BM25 + Vector Search
- Reranker 重排序
- Query Rewrite
- 多路召回
- 对话引用高亮
- RAGAS / Recall@K 等 RAG 评测指标
- 数据库持久化
- 用户权限与知识库隔离
- API 服务化部署

---

## 总结

该项目实现了一个完整的基础 RAG 链路：

```text
文档解析
→ Chunk 切分
→ Embedding
→ FAISS 向量索引
→ Top-K 语义检索
→ Prompt 构建
→ Qwen 回答生成
→ 来源展示
```

适合作为 RAG、企业知识库、技术文档问答以及大语言模型应用开发的入门与实践项目。
