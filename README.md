# 从零开始的机器学习：从算法到 AI 系统

[中文](README.md) | [English](README.en.md)

这是一门写给机器学习初学者的公开课程。它从“样本、特征和标签是什么”讲起，逐步进入线性模型、概率模型、树模型、神经网络、Transformer、推荐系统、强化学习和机器学习系统。

课程的算法主线参考 Stanford CS229 与吴恩达的 Machine Learning Specialization，工程部分吸收 Harvard CS249r 的系统视角。学习目标不只是调用模型，而是能回答三个问题：**为什么需要它、它怎样学到规律、它在什么情况下会失败。**

![欠拟合、合适拟合和过拟合](docs/assets/underfit-goodfit-overfit.png)

## 适合谁

- 第一次系统学习机器学习的本科生、研究生和工程师；
- 会基本电脑操作，但尚未学过 Python、线性代数或概率统计的学习者；
- 用过现成模型，希望补齐数学推导、诊断方法和系统知识的人；
- 想理解经典算法与 LLM、RAG、推荐系统之间联系的人。

课程不要求先修机器学习。零基础学习者从第 0 章开始即可；已有 Python 和数学基础的读者可以先做第 0 章自测，再按需跳读。

## 学完可以做什么

- 把一个真实问题写成样本、特征、标签、目标函数和评估指标；
- 理解并推导常见算法的目标函数、训练过程与关键假设；
- 用 NumPy 实现核心算法，再与 scikit-learn 或 PyTorch 的实现对照；
- 根据数据分布、残差、学习曲线和验证集表现判断欠拟合、过拟合与数据泄漏；
- 解释线性层、softmax、embedding、注意力、LoRA、向量检索等现代 AI 组件的来历；
- 完成一个包含基线、对照实验、误差分析和消融实验的课程项目。

## 从这里开始

1. [第 0 章：准备 Python、NumPy 与实验环境](docs/zh-CN/00-prerequisites.md)
2. [课程路线图：零基础 28 周 / 快速 24 周](docs/zh-CN/00-course-roadmap.md)
3. [第 1 章：什么是机器学习](docs/zh-CN/01-what-is-ml.md)
4. [附录 A：符号与默认约定](docs/zh-CN/appendix-a-notation.md)

如果你第一次接触机器学习，建议按章节顺序学习。第 1～3 章会反复使用“房租预测”这个例子，把数据、向量、损失函数、梯度下降和模型选择连成第一条完整算法链。

## 课程目录

| 模块 | 章节 | 要解决的问题 |
|---|---|---|
| 准备与基础 | [0. 学习准备](docs/zh-CN/00-prerequisites.md) · [1. 什么是机器学习](docs/zh-CN/01-what-is-ml.md) · [2. 数学基础](docs/zh-CN/02-math-foundations.md) | 怎样描述数据、任务、误差与学习过程？ |
| 监督学习 | [3. 线性回归](docs/zh-CN/03-linear-regression.md) · [4. 逻辑回归与 GLM](docs/zh-CN/04-logistic-regression.md) · [5. 生成式分类](docs/zh-CN/05-generative-classifiers.md) | 怎样预测数值、类别与概率？ |
| 相似度与集成 | [6. kNN 与核方法](docs/zh-CN/06-neighbors-and-kernels.md) · [7. 支持向量机](docs/zh-CN/07-support-vector-machines.md) · [8. 树与集成学习](docs/zh-CN/08-trees-and-ensembles.md) | 局部相似、最大间隔和规则组合各适合什么数据？ |
| 泛化与无监督学习 | [9. 学习理论与模型选择](docs/zh-CN/09-learning-theory-and-model-selection.md) · [10. 聚类、GMM 与 EM](docs/zh-CN/10-clustering-em-and-elbo.md) · [11. 降维](docs/zh-CN/11-dimensionality-reduction.md) | 怎样选择能泛化的模型，并从无标签数据中发现结构？ |
| 深度与表征学习 | [12. 神经网络与反向传播](docs/zh-CN/12-neural-networks-and-backprop.md) · [13. 表征学习](docs/zh-CN/13-representation-learning.md) · [14. CNN、RNN、LSTM 与 Attention](docs/zh-CN/14-sequence-vision-and-attention.md) | 网络怎样自动学习特征，并处理图像与序列？ |
| 现代生成与基础模型 | [15. Transformer 与 LLM](docs/zh-CN/15-transformers-llms-and-finetuning.md) · [16. VAE 与扩散模型](docs/zh-CN/16-vae-and-diffusion.md) | 大模型怎样训练、微调、推理和生成内容？ |
| 应用与决策 | [17. 推荐系统](docs/zh-CN/17-recommender-systems.md) · [18. 强化学习](docs/zh-CN/18-reinforcement-learning.md) | 怎样从海量物品中推荐，并优化连续决策？ |
| 评估、系统与项目 | [19. 评估、因果陷阱与 MLOps](docs/zh-CN/19-evaluation-causality-and-mlops.md) · [20. 机器学习系统](docs/zh-CN/20-machine-learning-systems.md) · [21. 面试与课程项目](docs/zh-CN/21-interview-and-capstone.md) | 怎样把离线模型变成可靠、快速、可维护的系统？ |

## 学习地图

```mermaid
flowchart LR
  A[问题与数据] --> B[数学基础]
  B --> C[经典监督学习]
  C --> D[泛化与无监督学习]
  D --> E[神经网络与表征学习]
  E --> F[Transformer与生成模型]
  F --> G[推荐与强化学习]
  G --> H[评估、部署与系统]
  H --> I[课程项目]
```

算法章节通常沿着同一条路径展开：

> 问题从哪里来 → 观察数据与现象 → 定义模型 → 推导目标函数 → 手算小例子 → 编写代码 → 分析失败案例 → 联系现代 AI → 自测与练习

建议先读图和例子，再进入公式；读完推导后关掉页面，尝试用自己的话解释每个符号。代码不要只运行一遍：改一个数据点、一个超参数或一个模型假设，再观察图和指标怎样变化。

## 安装与运行

需要 Python 3.10 或更高版本。第一次使用时，在终端运行：

```bash
git clone https://github.com/lieming358/machine-learning-from-zero.git
cd machine-learning-from-zero
python -m venv .venv
python -m pip install -r requirements-course.txt
python code/chapter00/readiness_check.py
python code/chapter00/first_workflow.py
```

虚拟环境的激活方法和常见报错见[第 0 章](docs/zh-CN/00-prerequisites.md)。各章代码位于 `code/chapterXX/`。例如：

```bash
python code/chapter03/regression_family_comparison.py
python code/chapter12/miniautograd.py
python code/chapter14/expanded_models.py
python code/chapter15/transformers_and_llms.py
```

图片和实验使用固定随机种子，可以在本地重新生成。提交修改前可运行课程检查：

```bash
python scripts/validate_course.py
```

## 仓库结构

```text
docs/zh-CN/       中文课程正文与答案
docs/en/          English chapters and solutions
docs/assets/      课程图解
code/chapterXX/   各章可运行示例
scripts/          课程检查工具
requirements-course.txt
```

每个中文章节顶部都有英文版本链接，英文页面也可以跳回中文。公式使用 GitHub 支持的 LaTeX，流程图使用 Mermaid，可以直接在 GitHub 阅读。

## 课程脉络

- [Stanford CS229](https://cs229.stanford.edu/)：监督学习、生成模型、学习理论、无监督学习与强化学习；
- [Andrew Ng Machine Learning Specialization](https://www.deeplearning.ai/specializations/machine-learning)：适合初学者的直觉、实践顺序与学习方法；
- [Harvard CS249r / Machine Learning Systems](https://github.com/harvard-edge/cs249r_book)：计算、内存、部署与软硬件协同视角。

本课程根据上述知识脉络重新组织并独立编写，与这些课程及其所属机构没有隶属关系。

## 反馈与勘误

如果你发现公式、代码、图片或链接有问题，欢迎提交 Issue。提问时请附上章节、运行命令、完整报错和你的理解；这些信息通常比一句“运行不了”更容易定位问题。
