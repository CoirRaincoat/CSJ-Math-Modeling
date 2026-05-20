"""生成 5.1 问题一 论文段落的 PDF"""
from fpdf import FPDF
import os

class PaperPDF(FPDF):
    def __init__(self):
        super().__init__('P', 'mm', 'A4')
        self.set_auto_page_break(True, 20)
        # Microsoft YaHei
        font_path = r'C:\Windows\Fonts\msyh.ttc'
        self.add_font('CN', '', font_path, uni=True)
        self.add_font('CN', 'B', r'C:\Windows\Fonts\msyhbd.ttc', uni=True)

    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font('CN', '', 8)
        self.cell(0, 10, f'{self.page_no()}', align='C')

    def title1(self, txt):
        self.set_font('CN', 'B', 14)
        self.multi_cell(0, 8, txt)
        self.ln(2)

    def title2(self, txt):
        self.set_font('CN', 'B', 12)
        self.multi_cell(0, 7, txt)
        self.ln(1)

    def title3(self, txt):
        self.set_font('CN', 'B', 11)
        self.multi_cell(0, 6.5, txt)
        self.ln(1)

    def body(self, txt):
        self.set_font('CN', '', 10.5)
        self.multi_cell(0, 6, txt, align='J')
        self.ln(0.5)

    def formula(self, txt):
        self.set_font('CN', '', 10)
        self.cell(0, 7, txt, align='C')
        self.ln(5)

    def table_caption(self, txt):
        self.set_font('CN', 'B', 9)
        self.cell(0, 6, txt, align='C')
        self.ln(7)

    def image_placeholder(self, txt):
        self.set_font('CN', '', 9)
        self.cell(0, 6, txt, align='C')
        self.ln(3)
        self.set_fill_color(240, 240, 240)
        self.set_draw_color(180, 180, 180)
        self.rect(self.get_x()+30, self.get_y(), 130, 50, 'DF')
        self.set_xy(self.get_x()+30, self.get_y()+18)
        self.set_font('CN', '', 10)
        self.cell(130, 10, '[ 此处插入对应图表 ]', align='C')
        self.ln(55)
        self.cell(0, 5, txt, align='C')
        self.ln(5)

    def add_table_row(self, cells, widths, bold=False):
        self.set_font('CN', 'B' if bold else '', 8)
        h = 6
        for i, (cell, w) in enumerate(zip(cells, widths)):
            self.cell(w, h, str(cell), border=1, align='C')
        self.ln()

pdf = PaperPDF()
pdf.add_page()

# ===== 5.1 问题一模型建立与求解 =====
pdf.title1('五、模型建立与求解')
pdf.title2('5.1 问题一模型建立与求解')
pdf.title3('5.1.1 数据预处理')
pdf.body('数据预处理是对原始数据进行清洗、转换和处理的过程，旨在提高数据质量、减少噪声和异常值的影响、使数据适合后续建模分析。本节从数据集概述和数据清洗两部分对数据进行预处理。')

pdf.title3('(1) 原始数据集概述')
pdf.body('本文使用的数据来源于杭州爱慷数食公司旗下某自助量贩餐厅的真实运营记录，包含三个附件文件。附件1为"餐厅销售流水信息表"，记录了每笔订单的消费时间、金额及营养汇总；附件2为"部分消费订单菜品具体信息表"，记录了部分订单中各菜品的名称、重量、单价和营养成分；附件3为数据字段说明文档。经检查发现，附件1和附件2的XLSX文件中均包含多个数据Sheet，须遍历加载全部Sheet后拼接为完整数据集。附件1共3个Sheet（indent_1~indent_3），附件2共15个Sheet（indent_details_1~indent_details_15）。全量加载后的数据集概况见表1。')

pdf.table_caption('表 1  数据集概况表')
widths1 = [50, 45, 45]
pdf.add_table_row(['指标', '附件1（流水信息）', '附件2（菜品详情）'], widths1, bold=True)
for row in [
    ['数据Sheet数', '3', '15'],
    ['总行数', '150,573', '72,129'],
    ['总列数', '17', '12'],
    ['唯一订单数', '149,626', '12,944'],
    ['时间跨度', '2022-09-02至2025-04-30', '—'],
    ['营业天数', '531', '—'],
    ['唯一菜品数', '—', '314'],
]:
    pdf.add_table_row(row, widths1)
pdf.ln(3)

pdf.body('附件1中各数值型字段的描述统计量见表2。')
pdf.table_caption('表 2  附件1数值型字段的描述统计量')
widths2 = [33, 20, 15, 15, 15, 18, 18]
pdf.add_table_row(['字段名称', '类型', '均值', '中位数', '标准差', '最大值', '最小值'], widths2, bold=True)
for row in [
    ['consume_money(元)', '连续', '11.24', '10.42', '7.39', '1280.98', '0.01'],
    ['calories(kcal)', '连续', '716.0', '680.6', '348.8', '47,237.9', '0.5'],
    ['protein(g)', '连续', '38.5', '34.9', '22.4', '2,030.7', '0.0'],
    ['fat(g)', '连续', '25.9', '22.1', '19.1', '2,928.6', '0.0'],
    ['carbohydrates(g)', '连续', '80.5', '72.6', '38.1', '4,565.5', '0.0'],
    ['fiber(g)', '连续', '6.0', '5.4', '3.2', '431.4', '0.0'],
]:
    pdf.add_table_row(row, widths2)
pdf.ln(3)

pdf.body('通过表1和表2可以看出，附件1的consume_money字段反映每笔订单的消费金额，平均客单价为11.24元；附件1中每条记录还汇总了该订单的总热量、蛋白质、脂肪、碳水化合物和膳食纤维含量。附件2的indent_id字段与附件1关联，但仅覆盖了约8.7%的订单（12,944/149,626）。dish_name为菜品名称，共有314种唯一菜品。weight字段为单份菜品的重量（单位：克），unit_price为对应单价。')

pdf.title3('(2) 数据清洗')
pdf.body('数据清洗确保数据的质量、准确性和一致性，为后续分析和建模提供可靠基础。')
pdf.body('对附件1进行缺失值检测，发现wallet_id、card_serial、user_phone_number、qr_code四个字段在所有150,573条记录中均为空值。这四个字段属于用户身份信息字段，对后续分析无贡献，予以删除。')
pdf.body('对附件1中消费金额和四项营养素字段进行异常值检测。采用IQR方法，计算各字段在1%-99%分位数区间外的潜在异常记录数，结果见表3。')

pdf.table_caption('表 3  附件1异常值检测结果（IQR，1%-99%分位数）')
widths3 = [30, 18, 18, 18, 14]
pdf.add_table_row(['字段', 'Q1', 'Q3', '异常记录数', '占比'], widths3, bold=True)
for row in [
    ['consume_money', '2.7', '30.9', '3,005', '2.0%'],
    ['calories', '202.0', '1,676.0', '3,012', '2.0%'],
    ['protein', '8.1', '101.2', '3,011', '2.0%'],
    ['fat', '1.9', '75.0', '3,007', '2.0%'],
    ['carbohydrates', '11.6', '179.5', '3,011', '2.0%'],
]:
    pdf.add_table_row(row, widths3)
pdf.ln(3)

pdf.body('对于上述异常记录，考虑到自助量贩餐厅的经营特点——团体订餐或大食量顾客可能导致单笔高消费，这些"异常值"可能反映了真实的业务场景而非数据录入错误。因此本文选择保留全部记录（标记但不剔除），在后续建模中由模型自行处理。')
pdf.body('对附件2中total_price、weight、unit_price三个数值字段进行类型检查和转换，确保后续计算精度。对dish_name字段去除首尾空格，避免因空格差异将同一菜品统计为不同菜品。')
pdf.body('数据清洗完成后，按消费小时将交易记录划分为午餐（10:00-14:00）和晚餐（16:00-20:00）两个餐次。经统计，午餐时段占总交易记录的99.2%，晚餐仅占0.8%，表明该餐厅是以午餐为核心经营时段的量贩模式。')

# ===== 5.1.2 描述性统计分析 =====
pdf.title3('5.1.2 描述性统计分析')
pdf.body('基于清洗后的531天日级汇总数据，对餐厅的消费规律进行了系统的描述性统计分析。通过菜品销量排名与ABC分类，发现314种菜品呈现典型的长尾分布特征——约76种A类菜品贡献了80%的销量，而168种C类菜品仅贡献5%。')

pdf.image_placeholder('[ 图1  p1_sales_distribution.png ]\n图1  菜品销售量分布图（销量Top20 / 销售额Top20 / ABC分析 / 类别占比）')

pdf.body('对时间维度的分析显示，餐厅日均接待282人，日均销售额3,187元，客单价11.39元。工作日日均订单量（约286单）高于周末（约255单），采用Welch\'s t检验（不假设方差齐性）验证了两者存在极显著差异（t=6.35, p<0.001）。')

pdf.image_placeholder('[ 图2  p1_temporal_patterns.png ]\n图2  时间维度销售规律（日订单趋势 / 星期箱线图 / 月度趋势 / 周/末对比）')

pdf.body('午餐与晚餐的差异分析表明，晚餐消费金额中位数略低于午餐（10.42元 vs 10.51元），但晚餐每单的平均营养素摄入量高于午餐。考虑到晚餐仅占0.8%的订单量，晚餐的统计数据（n≈1,173）样本量偏少，仅供参考。')

pdf.image_placeholder('[ 图3  p1_meal_comparison.png ]\n图3  午餐vs晚餐对比（消费分布 / 时段分布 / 营养摄入）')

pdf.body('人均营养分析显示，日均热量摄入约721 kcal/人，脂肪供能比约32.9%，略高于《中国居民膳食指南（2022）》推荐的20%-30%上限，提示餐厅菜品结构存在一定的油脂偏高问题。营养素之间的相关性矩阵显示，客单价与热量的相关性为0.65，与蛋白质的相关性为0.58，表明顾客消费金额越高，倾向于选择更多高热量、高蛋白的荤菜。')

pdf.image_placeholder('[ 图4  p1_nutrition_analysis.png ]\n图4  营养摄入分析（营养趋势 / 热量来源 / 相关矩阵 / 客单价分布）')

# ===== 5.1.3 关联规则挖掘 =====
pdf.title3('5.1.3 关联规则挖掘')
pdf.body('关联规则挖掘旨在从大量交易数据中发现菜品之间的隐含共购关系。本文采用Apriori算法对附件2中12,944个有菜品明细的订单进行关联规则挖掘。')
pdf.body('首先，将附件2的交易明细转换为购物篮二值矩阵。以订单编号为行索引，菜品名称为列索引，构建"交易-菜品"的0/1矩阵。过滤出现次数低于50次的低频菜品后，保留223种高频菜品。此过滤步骤有效降低了数据稀疏性，避免了过于罕见的菜品对频繁项集挖掘产生噪声干扰。')

pdf.body('关联规则度量指标。设A和B为两个菜品集合，定义三个核心指标：')
pdf.body('支持度（Support）：反映规则在全部交易中出现的频率；置信度（Confidence）：反映在A出现的条件下B也出现的概率；提升度（Lift）：衡量A的出现对B出现概率的提升程度，大于1表示正相关，等于1表示独立，小于1表示负相关。')

pdf.body('多级阈值挖掘策略。由于菜品销售频率极不均匀——米饭出现在93%的订单中，大量菜品频率低于1%——单一min_support阈值难以平衡。本文采用三级阈值搜索：依次设置min_support为0.01、0.005、0.003，最大项集长度设为3。在min_support=0.01时获得308个频繁项集（size=1: 145个，size=2: 148个，size=3: 15个），满足后续规则生成需求。')
pdf.body('筛选confidence≥0.25且lift≥1.15的规则，最终获得19条有意义的菜品关联规则。提升度最高的5条均为"米饭-酱鸭-豆芽/木耳"三元素组合（lift=8.51-8.78）。此外，"酸辣土豆丝→酱鸭腿"（lift=1.92）、"酱爆杏鲍菇→酱鸭腿"（lift=1.86）等规则揭示了素菜与荤菜的搭配偏好——酱鸭腿为该餐厅的"枢纽型"荤菜，与多达12种菜品形成稳定关联。')

pdf.table_caption('表 4  关联规则 Top 10')
widths4 = [10, 38, 34, 18, 18, 18]
pdf.add_table_row(['排名', '前件', '后件', '支持度', '置信度', '提升度'], widths4, bold=True)
for row in [
    ['1', '{米饭, 酱鸭}', '{豆芽/木耳}', '0.0103', '0.285', '8.78'],
    ['2', '{豆芽/木耳}', '{米饭, 酱鸭}', '0.0103', '0.316', '8.78'],
    ['3', '{米饭, 豆芽/木耳}', '{酱鸭}', '0.0103', '0.338', '8.70'],
    ['4', '{酱鸭}', '{米饭, 豆芽/木耳}', '0.0103', '0.265', '8.70'],
    ['5', '{豆芽/木耳}', '{酱鸭}', '0.0107', '0.330', '8.51'],
    ['6', '{酱鸭}', '{豆芽/木耳}', '0.0107', '0.277', '8.51'],
    ['7', '{酸辣土豆丝}', '{米饭, 酱鸭腿}', '0.0151', '0.305', '2.01'],
    ['8', '{米饭, 酸辣土豆丝}', '{酱鸭腿}', '0.0151', '0.325', '1.97'],
    ['9', '{酸辣土豆丝}', '{酱鸭腿}', '0.0157', '0.316', '1.92'],
    ['10', '{酱爆杏鲍菇}', '{酱鸭腿}', '0.0109', '0.307', '1.86'],
]:
    pdf.add_table_row(row, widths4)
pdf.ln(3)

pdf.image_placeholder('[ 图5  p1_association_rules.png ]\n图5  关联规则散点图与菜品共现网络图')

pdf.body('关联规则稳定性检验。考虑到部分规则的support值较低（排名第1仅为0.0103，对应约133个订单），小样本可能导致规则统计不稳定。为此采用Bootstrap重采样方法检验可靠性：对原始购物篮进行500次有放回抽样，每次重新运行Apriori并记录基线规则是否被再次发现，定义生存率为500次中该规则被发现的次数占比。')
pdf.body('结果表明，19条基线规则中有9条Bootstrap生存率超过80%。这批高稳定性规则均以"酱鸭腿"为后件（"酸辣土豆丝→酱鸭腿"生存率100%，"扎肉→酱鸭腿"生存率100%）。值得注意的是，提升度最高的前6条"米饭-酱鸭-豆芽/木耳"三重规则反而未通过80%生存率——这一"高lift、低稳定性"的反差揭示了小样本高指标的统计脆弱性。')

pdf.image_placeholder('[ 图6  p1_bootstrap_rules.png ]\n图6  Bootstrap规则稳定性检验')

pdf.body('上述关联规则为问题四的套餐设计提供了数据驱动的菜品搭配依据：高稳定性规则可作为套餐组合的优先候选搭配。')

# Output
out_path = r'C:\Users\CoirRaincoat\PyCharmMiscProject\CSJ-MathModeling\output\section_5_1.pdf'
pdf.output(out_path)
print(f'PDF saved to: {out_path}')
