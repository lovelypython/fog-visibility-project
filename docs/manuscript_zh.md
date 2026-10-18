# 基于 CNN 图像特征与机器学习的雾演变预测统一框架

> **论文工作稿（依据现有仓库材料整理）**。文中区分了已有的合成透射率 CNN 分类结果、100 条图像派生代理序列和本次按 70/30 重算的三模型代理预测。代理序列没有可核验的连续物理时间戳，预测结果不能解释为真实道路能见度预报。原报告、代码和结果文件保留在 paper/、src/transmittance_beta/ 和 results/；本稿按照目标评价体系报告预测误差。研究区、原始图像、同步气象观测仍需补齐。参考文献共 17 篇，其中 12 篇发表于 2023 年及以后；投稿前须通过数据库核对完整书目信息和 EI 会议格式。

**目标会议**：EI 收录会议（人工智能技术方向）
**建议英文题目**：*Fog Evolution Forecasting Using a Unified CNN and Random Forest Framework*
**对比模型**：SVR、Random Forest、XGBoost。本文报告预测误差指标。

## 摘要

雾演变涉及雾图像表征和低能见度序列预测两个环节。本文整理了一个融合卷积神经网络（CNN）与机器学习回归器的研究框架，并依据仓库中现有材料开展实验。紧凑型 CNN 在 4,575 张合成透射率测试图上的分类准确率为 92.46%，Macro-F1 为 0.9248；该结果只代表合成图像分类。对 100 条图像派生的消光系数代理序列，本文按时间索引重新划分可用样本，使用过去 5 步特征构造 SVR、随机森林和 XGBoost 输入，并按 70%/30% 顺序切分。29 条代理测试样本上，随机森林取得最低 MAE（19.95 km⁻¹）和 RMSE（23.96 km⁻¹），但由于序列的真实时间连续性和同步能见度观测未经验证，该结果仅作代理序列模型比较。本文不将该结果解释为真实道路预报。

**关键词**：雾演变；能见度预测；卷积神经网络；随机森林；支持向量回归；XGBoost

## 一、引言

雾是近地面水汽凝结形成的复杂天气现象，会降低水平能见度并影响交通运输和公共安全。雾的发生、维持及消散过程具有明显的时空变化特征，受相对湿度、温度、风速、地表条件和边界层结构等因素影响。随着自动气象站、摄像设备和遥感观测的普及，研究者能够获得逐时气象序列以及反映雾区空间纹理的图像数据，为数据驱动预测提供了多源信息基础。

现有数值预报和统计学习方法可用于天气及能见度预测，但雾的局地性和快速演变使预测仍面临困难。传统单变量时间序列方法主要利用历史数值关系，难以直接表示图像中的空间结构。端到端深度模型虽能学习复杂的时空映射，但通常需要大量标注样本和较高计算资源；在观测样本有限的业务场景中，如何有效融合视觉信息与表格特征仍值得研究。近年来，道路视觉方法已尝试从真实高速公路图像估计能见度，气象机器学习研究也关注机场能见度和低能见度事件预测[1–3]。数据驱动天气预报展示了深度学习对大尺度天气状态表征和快速推理的潜力[4–12]，但这些研究并不直接等同于局地雾预测，迁移有效性仍需实证检验。

针对上述问题，本文建立“合成图像 CNN 表征—真实图像指标—代理消光序列回归”的分阶段分析，并新增可接受同步图像和气象数据的预测代码。研究贡献是整理现有 CNN 训练证据，使用统一的滞后窗口和时间顺序切分比较 SVR、随机森林与 XGBoost，并将适用范围明确限制在当前代理序列。由于现有代理观测未含可验证的图像采集时间或同步能见度计标签，本稿尚不能声称已完成 CNN 嵌入与随机森林的端到端现场验证。

## 二、研究方法

### 2.1 图像特征提取模型：CNN

卷积神经网络通过局部卷积、非线性激活和下采样逐层学习图像特征。相比手工纹理指标，CNN 能够从像素中提取层次化空间表征；迁移学习可以复用大型图像数据上学习到的低层视觉滤波器，在有限样本任务中降低从头训练的需求；残差网络是常见的深层 CNN 架构[14]。现有研究材料包含一个用于 Foggy Cityscapes 合成透射率图的紧凑型 CNN：输入为灰度图，三层卷积通道为 1→8→16→24，随后自适应池化、Dropout 和全连接分类为 β=0.005、0.010、0.020 m⁻¹ 三类。训练损失为交叉熵，优化器为 AdamW。该分类器不是针对普通道路 RGB 照片训练的。新增代码另提供 ImageNet 预训练 ResNet-18 的冻结嵌入提取器，待获得获准使用且带时间戳的研究图像后用于主预测流程；它尚未在当前 100 条代理记录上运行。

对卷积输出使用 ReLU 激活，合成数据分类的训练目标为交叉熵：
\[
\mathcal{L}=-\sum_{c=1}^{3} y_c\log p_c.
\]
二维卷积为：
\[
 z_{i,j,k}=\sum_{u=1}^{K_h}\sum_{v=1}^{K_w}\sum_c W_{u,v,c,k}x_{i+u,j+v,c}+b_k,\qquad
 a_{i,j,k}=\max(0,z_{i,j,k}).
\]
残差块为：
\[
 \mathbf{y}=\mathcal{F}(\mathbf{x};\mathbf{W})+\mathbf{x}.
\]
合成 CNN 流程为：读取透射率 PNG 与文件名 β 标签 → 转灰度并缩放至 32×64 → 三层卷积分类 → 验证集选取最佳轮次 → 独立测试集评估。新增 ResNet 图像提取脚本的流程为：时间戳校验与 RGB 解码 → 预训练权重对应的缩放和归一化 → 前向推理 → 512 维嵌入导出。后者图像嵌入用于未来有时间戳的输入图像，必须使用历史滞后特征，不得取预测时点之后图像。

### 2.2 趋势预测模型：随机森林

随机森林通过对训练样本进行自助抽样，并在树节点随机抽取候选特征，训练多棵决策树后对回归结果取平均[15]。该方法能刻画非线性关系和变量交互，对输入尺度不敏感，且可由树的分裂收益形成变量重要性估计。设第 b 棵回归树为 T_b：
\[
 \hat{y}(\mathbf{x})=\frac{1}{B}\sum_{b=1}^{B}T_b(\mathbf{x}).
\]
节点分裂以降低节点内平方误差为目标。项目初始参数为树数 500、叶节点最小样本数 2、候选特征比例 0.8，并固定随机种子；最终参数须在论文中按实际代码及数据规模报告。流程为：训练样本自助抽样 → 随机选择分裂特征 → 生长回归树 → 对树预测求平均。

### 2.3 研究内容与技术路线

数据收集（雾图像、能见度、气象观测）→时间戳及单位核验→图像 RGB/尺寸预处理→预训练 ResNet-18 CNN 嵌入→表格变量按时间排序并构造过去 1–5 时次滞后值→按时间戳合并数据→清理缺失历史窗口/目标/图像嵌入→按时间顺序 70%/30% 切分→只用训练集拟合标准化→SVR、Random Forest、XGBoost 训练→测试期滚动一步预测→MSE、MAE、RMSE 和图表。

### 2.4 特征与预测评价指标

CNN 特征为 CNN 输出向量各维嵌入值；对嵌入不声称具有直接物理意义。表格特征包括研究数据实际提供的能见度、相对湿度、气温、风速、气压等变量的滞后值。标准化变换为 \(z=(x-\mu_{train})/\sigma_{train}\)，其中均值与标准差只由训练集估计。SVR 对输入和目标均在训练集拟合标准化变换；树模型对输入也采用训练集拟合的标准化预处理。

对测试集 n 个观测 y_i 与预测 ŷ_i，报告：
\[
 \mathrm{MSE}=\frac1n\sum_{i=1}^n(y_i-\hat y_i)^2,\quad
 \mathrm{MAE}=\frac1n\sum_{i=1}^n|y_i-\hat y_i|,\quad
 \mathrm{RMSE}=\sqrt{\frac1n\sum_{i=1}^n(y_i-\hat y_i)^2}.
\]
MSE 对较大误差惩罚更强；MAE 表示平均绝对误差；RMSE 与目标变量具有相同量纲。三项指标均越低表示测试误差越小。若另行构造雾/非雾分类任务，应补充类别阈值、混淆矩阵及分类指标，不能将回归误差替代分类评价。

## 三、图像特征提取

### 3.1 数据来源

现有项目资料包含四类数据：Foggy Cityscapes 合成透射率图及其 β 文件名标签；100 张真实雾图像的派生视觉指标和 TOPSIS 代理值；100 帧高速公路截图的相对视觉序列；以及两个事件时段共 2,878 条有效分钟级 AMOS 能见度与气象记录。主对比实验只使用 results/real_image_proxy_observations.csv 中 100 条图像派生记录；该表没有经过验证的物理时间戳。当前原始实景帧和 AMOS 原始观测未随主实验一并用于 CNN+回归融合。Foggy Cityscapes 数据本身不纳入仓库，须遵循其许可条款。

### 3.2 数据特征提取（CNN）

原研究的合成透射率 CNN 测试集包含 4,575 张平衡样本（每类 1,525 张），准确分类 4,230 张，准确率 92.46%，Macro-F1 为 0.9248，交叉熵为 0.23637。混淆错误出现在相邻 β 类之间。此结果用于验证合成域中的类别识别能力，不能替代真实道路图像和能见度计的配对验证。另有 100 条真实图像派生特征记录（亮度、对比度、饱和度、暗通道、边缘密度及 TOPSIS 严重度分数），但原始帧未与同步物理标签一并提供。新增的 ResNet-18 嵌入脚本尚未对这些记录生成嵌入，因为原图不可用。

### 3.3 指标合成（评价方法结果）

现有图像严重度分数 C_i 经项目既定映射构造代理衰减系数：β_proxy=3.912+74.328C_i（km⁻¹），再按 V_proxy=3912/β_proxy（m）转换为代理能见度。该换算仅为项目代理尺度，不是用同步实测能见度校准的物理量。原研究以熵权、AHP 和 TOPSIS 合成亮度、对比度、饱和度、暗通道及边缘密度指标；100 条记录的 β_proxy 用于本次回归目标。合成结果与真实能见度计校准仍待完成。

## 四、实证预测结果

### 4.1 数据可视化与描述性统计

对构造五步滞后后剩余的 95 条记录进行描述性统计，指标为代理序列，不应解释为已知频率的物理时间序列。

| 变量 | 均值 ± 标准差 | 最小值–最大值 |
|---|---:|---:|
| 代理消光系数 (km⁻¹) | 48.19 ± 21.67 | 3.91–78.24 |
| 亮度 | 0.765 ± 0.107 | 0.443–0.937 |
| 对比度 | 55.04 ± 14.74 | 30.58–88.65 |
| 饱和度 | 0.061 ± 0.066 | 0.000–0.311 |
| 暗通道 | 0.748 ± 0.120 | 0.382–0.934 |
| 边缘密度 | 0.200 ± 0.001 | 0.195–0.200 |
| TOPSIS 严重度分数 | 0.596 ± 0.292 | 0.000–1.000 |

数据的边缘密度变化很小，不能据此解读为普遍物理规律。由于观测索引没有可信时间间隔，本研究不把该序列画成真实分钟/小时雾演变曲线；图中只按记录索引展示 29 个留出样本。程序也为新数据生成逐时预测、描述性统计和误差图。

### 4.2 数据预处理

按时间升序排列，去除重复时间戳（保留规则须与数据源一致），构造过去五个时次的滞后特征。将可用样本前 70% 作为训练集、后 30% 作为测试集。SVR 采用仅在训练集拟合的标准化变换；相同训练集统计量应用于测试集。随机森林和 XGBoost 输入也采用训练集拟合的标准化流程。所有模型按滚动一步方式在测试时段预测：预测时仅使用该时刻之前已观测的滞后变量。应报告切分日期、样本数、缺失处理和目标单位。

### 4.3 SVR、Random Forest 与 XGBoost 预测结果（核心二）

模型预测目标为既有数据中的 beta_proxy_km-1，即基于 100 张图像派生指标计算的代理消光系数（km⁻¹），不是仪器实测消光系数。输入包含该代理目标以及亮度、对比度、饱和度、暗通道、边缘密度、严重度分数各自过去 5 个观测索引的值。排序索引并非经验证的物理时间轴，因此本实验是按原始排列顺序的代理序列基准。删除前五个无法形成完整滞后窗口的样本后，可用 95 条记录，前 66 条训练，后 29 条测试（等于可用样本的约 70/30）。

SVR[16] 使用 RBF 核（C=10、epsilon=0.1）；随机森林[15] 使用 500 棵树、叶节点最小样本数 2、候选特征比例 0.8；XGBoost[17] 使用 500 棵树、最大深度 5、学习率 0.03、行/列采样率 0.8、lambda=1。所有标准化参数仅在训练集拟合。因代理消光系数按物理含义非负，测试预测下限截断为 3.912 km⁻¹；该约束仅应用于此代理配置。测试集结果如下：

| 模型 | MSE (km⁻²) | MAE (km⁻¹) | RMSE (km⁻¹) | 测试样本数 |
|---|---:|---:|---:|---:|
| SVR | 971.23 | 24.98 | 31.16 | 29 |
| Random Forest | 574.20 | 19.95 | 23.96 | 29 |
| XGBoost | 647.75 | 21.30 | 25.45 | 29 |

在该小样本代理序列基准中，随机森林的三项误差均低于 SVR 和 XGBoost。对应的真实值/预测值、误差图、描述性统计和 XGBoost 特征重要性保存在 results/proxy_forecast_70_30/。该结果不能证明森林对真实雾演变最佳：样本排列是否等同连续时间未验证，序列短且由图像指标派生，没有同步路侧能见度观测；而且这组输入并未包含本研究 ResNet-18 的图像嵌入。真实图像 CNN 特征融合实验须待原始时间戳图像和对应观测补齐后运行。

## 五、结论与展望

### 研究内容
本文建立了以预训练 CNN 图像嵌入和滞后气象序列为输入、以 SVR/随机森林/XGBoost 为预测器的雾演变预测实验流程。

### 研究结论
现有证据显示，合成透射率图上的 CNN 分类效果达到 92.46% 准确率和 0.9248 Macro-F1；100 条图像派生代理序列上的小样本对比中，随机森林误差最低。两项结果的输入域和目标不同，且代理序列不具备经过验证的物理时间和同步能见度标签，因此不能据此宣称已经实现真实道路雾演变预报。下一阶段需要在同一批按时间同步的道路图像和能见度/气象观测上提取 CNN 特征，开展滚动时间验证后再评估融合模型。

### 未来研究方向
未来可扩大跨区域、多季节样本并开展外部站点验证；进一步研究 CNN 微调、时空网络及不确定性预测，并通过消融实验量化图像信息的增益。

## 参考文献（16 篇；2023 年及以后文献 11 篇）

[1] Xiao P, et al. Highway Visibility Estimation in Foggy Weather via Multi-Scale Fusion Network. *Sensors*. 2023;23(24):9739.

[2] Yang W, et al. Multi Visual Feature Fusion Based Fog Visibility Estimation for Expressway Surveillance Using Deep Learning Network. *Expert Systems with Applications*. 2023;234:121151.

[3] Penov N, Guerova G. Sofia Airport Visibility Estimation with Two Machine-Learning Techniques. *Remote Sensing*. 2023;15(19):4799.

[4] Lam R, et al. Learning skillful medium-range global weather forecasting. *Science*. 2023;382(6677):1416–1421. doi:10.1126/science.adi2336.

[5] Bi K, et al. Accurate medium-range global weather forecasting with 3D neural networks. *Nature*. 2023;619:533–538. doi:10.1038/s41586-023-06185-3.

[6] Nguyen T, et al. ClimaX: A foundation model for weather and climate. In: *Proceedings of the 40th International Conference on Machine Learning*. PMLR 202:5458–5474; 2023.

[7] Rasp S, et al. WeatherBench 2: A benchmark for the next generation of data-driven global weather models. *Journal of Advances in Modeling Earth Systems*. 2024;16:e2023MS004019. doi:10.1029/2023MS004019.

[8] Kochkov D, et al. Neural general circulation models for weather and climate. *Nature*. 2024;632:1060–1066. doi:10.1038/s41586-024-07744-y.

[9] Price I, et al. Probabilistic weather forecasting with machine learning. *Nature*. 2025. doi:10.1038/s41586-024-08252-9.

[10] Lang S, et al. AIFS—ECMWF's data-driven forecasting system. ECMWF Technical Memorandum; 2024.

[11] Bodnar C, et al. Aurora: A foundation model for the Earth system. *arXiv preprint*; 2024. arXiv:2405.13063.

[12] Fu X, et al. FuXi: A cascade machine learning forecasting system for 15-day global weather forecast. *npj Climate and Atmospheric Science*. 2023;6:190. doi:10.1038/s41612-023-00512-1.

[13] Nayar SK, Narasimhan SG. Vision in bad weather. In: *Proceedings of the IEEE International Conference on Computer Vision*. 1999:820–827.

[14] He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2016:770–778. doi:10.1109/CVPR.2016.90.

[15] Breiman L. Random forests. *Machine Learning*. 2001;45:5–32. doi:10.1023/A:1010933404324.

[16] Cortes C, Vapnik V. Support-vector networks. *Machine Learning*. 1995;20:273–297. doi:10.1007/BF00994018.

[17] Chen T, Guestrin C. XGBoost: A scalable tree boosting system. In: *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. 2016:785–794. doi:10.1145/2939672.2939785.

> 注：以上共 16 篇，其中 2023 年及以后文献 11 篇（68.75%）。直接讨论雾/能见度的近年来源为 [1]–[3]；大尺度数据驱动天气预测文献只作为 AI 方法背景，不能替代雾场景验证。投稿前须通过数据库核对完整作者、页码、卷期和 DOI，并核验会议对 2023 年是否计入“近三年”的口径。
