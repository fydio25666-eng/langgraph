"""Evaluate the customer-support graph with a model-based judge.

Run from the project root after configuring BASE_URL, API_KEY and MODEL::

    python tests/eval_model.py

The judge uses the same configured ChatOpenAI client as the workflow.  Each
case gets a fresh LangGraph thread so that one test cannot affect another.
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

# Allow ``python tests/eval_model.py`` to import modules in the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import model  # noqa: E402
from graph import app as workflow_app  # noqa: E402
from schemas import ServiceReply  # noqa: E402


class EvalCase:
    """One customer question and the answer expected from the product manual."""

    def __init__(self, question: str, standard_answer: str) -> None:
        self.question = question
        self.standard_answer = standard_answer


class JudgeResult(BaseModel):
    """Structured rubric returned by the judge model."""

    accuracy: int = Field(ge=1, le=5, description="事实准确度，5分为完全正确")
    faithfulness: int = Field(
        ge=1, le=5, description="忠诚度/低幻觉程度，5分为无臆造且遵守已知信息"
    )
    relevance: int = Field(ge=1, le=5, description="回答相关度，5分为直击问题")
    tone_compliance: int = Field(
        ge=1, le=5, description="语气合规度，5分为专业、礼貌且安全"
    )
    comment: str = Field(min_length=1, max_length=200)


# The expected answers are deliberately concise.  The judge should reward the
# same facts expressed in different wording, rather than exact string matches.
CASES = [
    EvalCase("现货商品什么时候发货？", "每日16:30前付款的现货订单当天打包出库，16:30后次日发货。"),
    EvalCase("满多少元包邮？偏远地区也包邮吗？", "常规商品实付满99元包邮；新疆、西藏、青海、内蒙古、宁夏、海南等偏远地区除外。"),
    EvalCase("店里默认用什么快递？", "常规小件默认中通或圆通；重货/批量订单可能使用德邦、安能或专线汽运。"),
    EvalCase("可以开增值税专用发票吗？", "可以。专票需单笔订单满500元或累计开具；确认收货后提供抬头、统一社会信用代码、开户行账号和收票邮箱，5个工作日内开出。"),
    EvalCase("定制打孔的配件能七天无理由退货吗？", "不能。非标尺寸、专属打孔、特定长度切割等已投产定制品不支持七天无理由退货，重大加工质量缺陷除外。"),
    EvalCase("收到货发现零件少了怎么办？", "签收时验货并拍照/录像留证，直接拒签或在面单备注异常签收，签收后24小时内联系客服申请补发或赔付。"),
    EvalCase("轻钢龙骨五金配件有哪些？", "主要有主龙骨吊件、副龙骨挂件、膨胀吊杆套件、边龙骨连接件、对接插片、穿线孔护套管卡和十字加强连接件。"),
    EvalCase("产品的防锈材质和工艺是什么？", "采用热镀锌带钢冷弯冲压成型，双面镀锌层按国标优等品不低于120g/㎡，适合潮湿环境防腐。"),
    EvalCase("50主龙骨吊件能承重多少？", "50主龙骨吊件单点安全承重达80kg。"),
    EvalCase("非标冲压打孔定制有什么要求？", "支持定制；需提供CAD图纸或实体样品，通常单批起订1000套，公差可控制在±0.2mm以内。"),
    EvalCase("钢丝钳是什么材质，硬度多少？", "主体为CR-V铬钒合金钢，钳口和刃口经高频淬火，刃口硬度HRC 58-62。"),
    EvalCase("6寸和8寸钢丝钳剪切能力有什么区别？", "6寸适合铜/铝线不超过3.0mm、中硬铁丝不超过1.6mm；8寸适合中碳钢丝不超过2.5mm、硬质铁丝不超过2.0mm，禁止剪预应力钢绞线和高硬度弹簧钢丝。"),
    EvalCase("尖嘴钳可以带电作业吗？", "不可以。常规手柄耐压低于250V，不是专业绝缘工具，带电或高压作业应使用VDE 1000V认证绝缘钳。"),
    EvalCase("工具表面的防锈油要怎么处理？", "用干净抹布擦去浮油即可，不必用洗洁精清洗；存放在干燥通风处，长期闲置可在转轴和钳口滴2-3滴机油或WD-40。"),
    EvalCase("免打孔置物架适合什么墙面？", "适合光滑瓷砖、大理石、钢化玻璃、光滑金属板和实木复合板等坚硬平整墙面；白灰墙、石膏板乳胶漆墙、壁纸、掉灰或渗水墙面不能用胶固定。"),
    EvalCase("免打孔胶贴好后多久能放东西？", "清洁并干燥墙面后按压1-2分钟、贴辅助贴，必须静置满72小时；期间不能放物品或淋水。"),
    EvalCase("置物架免打孔安装最大承重是多少？", "在平整瓷砖墙面并严格固化72小时后，常规单层架静态安全承重约15-20kg。"),
    EvalCase("汽车顶胶怎么确认是否适配我的车？", "提供行驶证正面照片或17位VIN，由技术人员通过原厂EPC反查OE号、排量和出厂年月后匹配，不能只凭车型目测购买。"),
    EvalCase("更换减振顶胶后需要四轮定位吗？", "需要。施工会影响外倾角和前束角，应使用专业四轮定位仪校准，避免轮胎偏磨和高速方向发飘。"),
    EvalCase("顶胶质保多久？", "全系顶胶及总成质保1年或行驶20,000公里，以先到者为准；确认是非人为质量问题后免费以换代修。"),
]


def invoke_agent(case: EvalCase, index: int) -> str:
    """Run the real graph once and extract its customer-facing reply."""

    state = workflow_app.invoke(
        {"question": case.question},
        config={"configurable": {"thread_id": f"eval-{index}-{uuid4().hex}"}},
    )
    result = state.get("result")
    if isinstance(result, ServiceReply):
        return result.reply
    if result is not None and hasattr(result, "reply"):
        return str(result.reply)
    return "未生成有效回答"


def judge_answer(case: EvalCase, actual_answer: str) -> JudgeResult:
    """Ask the configured model to score one answer against the reference."""

    assert model is not None
    prompt = f"""你是严格、客观的电商五金客服质量评审员。
请对比【标准答案】与【Agent实际回答】，只输出符合要求的 JSON。
四项均为1-5整数：
- accuracy：事实准确度，是否覆盖标准答案关键事实；
- faithfulness：忠诚度/低幻觉程度，是否只使用已知信息，5分表示没有臆造；
- relevance：回答相关度，是否直接回答问题且无明显跑题；
- tone_compliance：语气合规度，是否礼貌、专业、清晰，并包含必要的安全提醒。
comment 用中文写一句不超过200字的简评。不要因为措辞不同而扣分，也不要补充标准答案之外的事实。

【标准答案】
{case.standard_answer}

【Agent实际回答】
{actual_answer}
"""
    raw = model.with_structured_output(JudgeResult, method="json_mode").invoke(prompt)
    return JudgeResult.model_validate(raw)


def main() -> None:
    if model is None:
        raise SystemExit(
            "未配置大模型，请先设置 BASE_URL、API_KEY、MODEL 后再运行 tests/eval_model.py。"
        )

    totals = {"accuracy": 0, "faithfulness": 0, "relevance": 0, "tone_compliance": 0}
    print(f"开始评测，共 {len(CASES)} 题\n")
    for index, case in enumerate(CASES, start=1):
        actual = invoke_agent(case, index)
        result = judge_answer(case, actual)
        scores = {
            "accuracy": result.accuracy,
            "faithfulness": result.faithfulness,
            "relevance": result.relevance,
            "tone_compliance": result.tone_compliance,
        }
        for key, value in scores.items():
            totals[key] += value
        average = sum(scores.values()) / 4
        print(f"[{index:02d}] {case.question}")
        print(
            "     准确度 {accuracy}/5 | 忠诚度/幻觉率 {faithfulness}/5 | "
            "回答相关度 {relevance}/5 | 语气合规度 {tone_compliance}/5 | "
            "本题平均 {average:.2f}/5".format(**scores, average=average)
        )
        print(f"     评价：{result.comment}")
        print(f"     Agent回答：{actual}\n")

    count = len(CASES)
    averages = {key: value / count for key, value in totals.items()}
    overall = sum(averages.values()) / 4
    print("=" * 72)
    print("最终平均分（1-5分，分数越高越好）")
    print(f"准确度：{averages['accuracy']:.2f}")
    print(f"忠诚度/幻觉率：{averages['faithfulness']:.2f}（忠诚度越高表示幻觉越少）")
    print(f"回答相关度：{averages['relevance']:.2f}")
    print(f"语气合规度：{averages['tone_compliance']:.2f}")
    print(f"综合平均：{overall:.2f}")


if __name__ == "__main__":
    main()
