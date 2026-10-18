# 基于 CNN 图像特征与机器学习的雾演变预测统一框架

> **论文工作稿（待真实数据填充）**。本稿依据当前代码框架组织，不代表已完成实验。仓库中没有原始雾图像、气象观测、数据来源说明或原论文材料；方括号内容须据实补全。文末文献为代表性基础文献，因环境网络代理拒绝文献检索，卷期页码和 DOI 尚未逐条核验。当前清单也未满足“近三年英文文献至少三分之二”的投稿要求，须补充至少 10 篇 2023–2026 年直接讨论雾、低云或能见度预测的英文论文。

**目标会议**：EI 收录会议（人工智能技术方向）  
**建议英文题目**：*Fog Evolution Forecasting Using a Unified Framework Combining CNN Image Feature Extraction and Machine Learning Models*  
**对比模型**：SVR、Random Forest、XGBoost。本文不使用 ARIMA，也不报告 R²。

## 摘要

雾的形成和消散会显著影响道路、航空及其他近地面活动。雾演变受气象条件和局地环境共同影响，单一数值序列难以完整表达雾区图像中的空间结构。本文拟构建一个融合卷积神经网络（CNN）图像表征与机器学习回归器的雾演变预测框架。首先对时间戳对齐的雾图像使用预训练 CNN 提取高维视觉嵌入；随后将嵌入与能见度及气象观测按时间对齐，构造过去 5 个时次的滞后特征，并按时间顺序划分训练集与测试集。在训练集上拟合标准化变换后，分别训练支持向量回归、随机森林和 XGBoost，对测试期能见度（或经明确定义的雾强度指标）进行滚动一步预测。模型以均方误差、平均绝对误差和均方根误差评价。最终性能、数据规模、研究区域及模型优劣需待实际数据运行后填写，本文不预设或虚构数值结果。

**关键词**：雾演变；能见度预测；卷积神经网络；随机森林；支持向量回归；XGBoost

## 一、引言

雾是近地面水汽凝结形成的复杂天气现象，会降低水平能见度并影响交通运输和公共安全。雾的发生、维持及消散过程具有明显的时空变化特征，受相对湿度、温度、风速、地表条件和边界层结构等因素影响。随着自动气象站、摄像设备和遥感观测的普及，研究者能够获得逐时气象序列以及反映雾区空间纹理的图像数据，为数据驱动预测提供了多源信息基础。

现有数值预报和统计学习方法可用于天气及能见度预测，但雾的局地性和快速演变使预测仍面临困难。传统单变量时间序列方法主要利用历史数值关系，难以直接表示图像中的空间结构。端到端深度模型虽能学习复杂的时空映射，但通常需要大量标注样本和较高计算资源；在观测样本有限的业务场景中，如何有效融合视觉信息与表格特征仍值得研究。近年来，数据驱动天气预报展示了深度学习对大尺度天气状态表征和快速推理的潜力[1–8]，但这些工作并不直接等同于局地雾预测，其方法能否迁移到雾演变问题仍需实证检验。

针对上述问题，本文拟建立统一的“图像表征—时间对齐—滞后建模—预测评价”流程：使用预训练 CNN 提取雾图像嵌入；将图像嵌入与地面观测配对；基于过去五个时次的图像和气象变量构建输入；比较 SVR、随机森林和 XGBoost 三种机器学习回归方法。本文的预期贡献包括：①构建可复现的雾图像特征提取与时间序列预测流程；②在严格按时间划分且只使用训练集拟合标准化参数的设定下比较三类回归器；③输出逐时预测、误差指标和特征重要性，便于分析图像特征与气象因子对预测的作用。上述贡献及有效性需以真实数据实验验证。

## 二、研究方法

### 2.1 图像特征提取模型：CNN

卷积神经网络通过局部卷积、非线性激活和下采样逐层学习图像特征。相比手工纹理指标，CNN 能够从像素中提取层次化空间表征；迁移学习可以复用大型图像数据上学习到的低层视觉滤波器，在有限样本任务中降低从头训练的需求[9–11]。本研究拟采用 ImageNet 预训练 ResNet-18，将最终全连接分类层替换为恒等映射，得到 512 维全局池化嵌入。当前实现冻结网络参数，不用本研究测试集图像训练 CNN；后续如有足量标注训练样本，可另行设计仅在训练期微调的消融实验。

二维卷积为：
\[
 z_{i,j,k}=\sum_{u=1}^{K_h}\sum_{v=1}^{K_w}\sum_c W_{u,v,c,k}x_{i+u,j+v,c}+b_k,\qquad
 a_{i,j,k}=\max(0,z_{i,j,k}).
\]
残差块为：
\[
 \mathbf{y}=\mathcal{F}(\mathbf{x};\mathbf{W})+\mathbf{x}.
\]
图像处理流程为：时间戳校验与 RGB 解码 → 按预训练权重要求缩放和归一化 → ResNet-18 前向推理 → 全局池化嵌入 → 按时间戳导出 CSV。图像嵌入在预测模型中使用滞后 1–5 个时次，避免把目标时点尚不可获得的图像用于预测。

### 2.2 趋势预测模型：随机森林

随机森林通过对训练样本进行自助抽样，并在树节点随机抽取候选特征，训练多棵决策树后对回归结果取平均[12]。该方法能刻画非线性关系和变量交互，对输入尺度不敏感，且可由树的分裂收益形成变量重要性估计。设第 b 棵回归树为 T_b：
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
MSE 对较大误差惩罚更强；MAE 表示平均绝对误差；RMSE 与目标变量具有相同量纲。三项指标均越低表示测试误差越小。本文按要求不使用 R²。若另行构造雾/非雾分类任务，应补充类别阈值、混淆矩阵及分类指标，不能将回归误差替代分类评价。

## 三、图像特征提取

### 3.1 数据来源

图像及气象数据来自[数据集/观测站全称]，观测区域为[经纬度/地理范围]，时间范围为[起止日期]，图像采样间隔为[间隔]。图像由[摄像机/卫星/其他传感器]采集，空间分辨率为[分辨率]；气象变量来自[站点/再分析产品/数据接口]。应报告数据许可、质量控制、时间同步容差、缺测比例、异常值处理和样本筛选规则。当前尚无上述数据，字段均待补。

### 3.2 数据特征提取（CNN）

按时间戳将每幅图像与观测记录关联，使用 ResNet-18 ImageNet 权重提取 512 维图像嵌入。报告实际输入图像数、成功处理数、尺寸变换及权重版本/校验信息。主实验使用冻结特征；训练、测试按时间隔离，且不根据测试表现选择图像处理方式。建议报告训练样本的 PCA 或 UMAP 二维可视化作为描述性分析，不把降维图误称为预测性能结果。

**结果待填**：可用图像数 [ ]；匹配成功率 [ ]%；每时次图像数 [ ]；图像质量控制规则 [ ]。未取得图像前不生成占位结果图。

### 3.3 指标合成（评价方法结果）

若研究目标为能见度回归，直接使用能见度（m）作为目标，并报告单位。若希望使用“雾演变指标”，须先给出可复现定义，例如基于本地业务标准将能见度映射为雾强度等级，说明阈值来源和边界处理；本文不自行指定未经数据源支持的阈值。多源变量如需综合为单一指标，应说明归一化方法、权重来源、方向一致性和敏感性分析。当前没有实际数据或已定义的综合指标，因此本节结果待补。

## 四、实证预测结果

### 4.1 数据可视化与描述性统计

运行程序后输出目标变量及输入变量的描述性统计表、测试期预测曲线和误差比较图。论文应补充全时段雾演变走势图，并在表格报告每变量样本量、均值、标准差、中位数、四分位数、最小值和最大值。对强偏态变量可补充分位数或分布图。当前没有真实数据，不能提供数值表或趋势解释。

### 4.2 数据预处理

按时间升序排列，去除重复时间戳（保留规则须与数据源一致），构造过去五个时次的滞后特征。将可用样本前 70% 作为训练集、后 30% 作为测试集。SVR 采用仅在训练集拟合的标准化变换；相同训练集统计量应用于测试集。随机森林和 XGBoost 输入也采用训练集拟合的标准化流程。所有模型按滚动一步方式在测试时段预测：预测时仅使用该时刻之前已观测的滞后变量。应报告切分日期、样本数、缺失处理和目标单位。

### 4.3 SVR、Random Forest 与 XGBoost 预测结果（核心二）

SVR 使用 RBF 核，初始参数 C=10、epsilon=0.1；随机森林使用 500 棵树、叶节点最小样本数 2；XGBoost 使用 500 棵树、最大深度 5、学习率 0.03、行/列采样率 0.8、lambda=1。这些是可复现的起始参数，不代表已经调优。正式实验应在训练期内部采用按时间顺序的交叉验证调参，避免用测试集选参，并报告软件版本与最终参数。

| 模型 | MSE | MAE | RMSE | 测试样本数 |
|---|---:|---:|---:|---:|
| SVR | 待运行 | 待运行 | 待运行 | 待运行 |
| Random Forest | 待运行 | 待运行 | 待运行 | 待运行 |
| XGBoost | 待运行 | 待运行 | 待运行 | 待运行 |

程序生成测试集真实值与三模型预测曲线、MAE/RMSE 对比图、逐时预测 CSV 和 XGBoost 特征重要性图。结果解读应围绕误差差异、雾快速生成/消散时的偏差、不同能见度区间误差及跨季节稳定性展开；取得实际输出后才可写“最佳模型”或研究结论。树模型内置重要性受特征相关性影响；如需解释相关变量贡献，建议补充仅在测试集上计算的置换重要性或 SHAP，并明确解释范围。

## 五、结论与展望

### 研究内容
本文建立了以预训练 CNN 图像嵌入和滞后气象序列为输入、以 SVR/随机森林/XGBoost 为预测器的雾演变预测实验流程。

### 研究结论
**待真实数据和模型输出后撰写。** 应据 MSE、MAE、RMSE 和测试曲线总结相对表现，说明适用时段与误差特征，不得依据训练集误差推断泛化能力。当前没有实验结果，不能形成实证结论。

### 未来研究方向
未来可扩大跨区域、多季节样本并开展外部站点验证；进一步研究 CNN 微调、时空网络及不确定性预测，并通过消融实验量化图像信息的增益。

## 参考文献（工作清单，投稿前逐条核验）

[1] Lam R, et al. Learning skillful medium-range global weather forecasting. *Science*. 2023;382(6677):1416–1421. doi:10.1126/science.adi2336.

[2] Bi K, et al. Accurate medium-range global weather forecasting with 3D neural networks. *Nature*. 2023;619:533–538. doi:10.1038/s41586-023-06185-3.

[3] Nguyen T, et al. ClimaX: A foundation model for weather and climate. In: *Proceedings of the 40th International Conference on Machine Learning*. PMLR 202; 2023.

[4] Rasp S, et al. WeatherBench 2: A benchmark for the next generation of data-driven global weather models. *Journal of Advances in Modeling Earth Systems*. 2024;16:e2023MS004019. doi:10.1029/2023MS004019.

[5] Kochkov D, et al. Neural general circulation models for weather and climate. *Nature*. 2024;632:1060–1066. doi:10.1038/s41586-024-07744-y.

[6] Price I, et al. Probabilistic weather forecasting with machine learning. *Nature*. 2025. doi:10.1038/s41586-024-08252-9.

[7] Lang S, et al. AIFS—ECMWF's data-driven forecasting system. ECMWF Technical Memorandum; 2024.

[8] Bodnar C, et al. Aurora: A foundation model for the Earth system. *arXiv preprint*; 2024. arXiv:2405.13063.

[9] Krizhevsky A, Sutskever I, Hinton GE. ImageNet classification with deep convolutional neural networks. In: *Advances in Neural Information Processing Systems*. 2012;25:1097–1105.

[10] He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2016:770–778. doi:10.1109/CVPR.2016.90.

[11] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition. In: *International Conference on Learning Representations*. 2015.

[12] Breiman L. Random forests. *Machine Learning*. 2001;45:5–32. doi:10.1023/A:1010933404324.

[13] Cortes C, Vapnik V. Support-vector networks. *Machine Learning*. 1995;20:273–297. doi:10.1007/BF00994018.

[14] Chen T, Guestrin C. XGBoost: A scalable tree boosting system. In: *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. 2016:785–794. doi:10.1145/2939672.2939785.

[15] Paszke A, et al. PyTorch: An imperative style, high-performance deep learning library. In: *Advances in Neural Information Processing Systems*. 2019;32.

[16] Pedregosa F, et al. Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*. 2011;12:2825–2830.

> 上述清单包含天气 AI 代表研究和算法基础，不构成已完成的雾预测文献综述；投稿前须核验全部书目信息，并补充近三年直接相关雾/能见度研究，以满足会议要求。
