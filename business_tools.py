import re
from typing import Optional, Tuple
from product_catalog import PRODUCTS


class BusinessTools(object):
    SIZE_KEYWORDS = ["尺码", "身高", "体重", "穿多大", "买多大", "xxl", "xl", "l", "m", "s"]
    STYLE_KEYWORDS = ["推荐", "穿搭", "搭配", "上新", "适合", "预算", "风格", "颜色"]
    POLICY_KEYWORDS = ["退货", "换货", "退款", "发货", "物流", "售后"]

    def detect_intent(self, query: str) -> list[str]:
        intents = []
        q = query.lower()
        if any(k in q for k in self.SIZE_KEYWORDS):
            intents.append("size")
        if any(k in q for k in self.STYLE_KEYWORDS):
            intents.append("style")
        if any(k in q for k in self.POLICY_KEYWORDS):
            intents.append("policy")
        return intents

    def _extract_height_weight(self, query: str) -> Tuple[Optional[int], Optional[int]]:
        height_match = re.search(r"(\d{3})\s*cm", query.lower())
        weight_match = re.search(r"(\d{2,3})\s*(kg|公斤|斤)", query.lower())

        height = int(height_match.group(1)) if height_match else None
        weight = None
        if weight_match:
            raw_weight = int(weight_match.group(1))
            unit = weight_match.group(2)
            weight = raw_weight if unit in ("kg", "公斤") else int(raw_weight / 2)
        return height, weight

    def recommend_size(self, query: str) -> str:
        height, weight = self._extract_height_weight(query)
        if not height or not weight:
            return "尺码建议：请补充身高(cm)和体重(kg/斤)，例如“170cm，65kg”。"

        if height <= 160 or weight <= 50:
            size = "S"
        elif height <= 168 or weight <= 58:
            size = "M"
        elif height <= 176 or weight <= 68:
            size = "L"
        elif height <= 184 or weight <= 80:
            size = "XL"
        else:
            size = "XXL"

        return f"尺码建议：根据你提供的 {height}cm / {weight}kg，建议优先试 {size} 码（不同版型可能上下浮动一码）。"

    def recommend_products(self, query: str) -> str:
        q = query.lower()
        budget_match = re.search(r"(\d{2,4})\s*元", q)
        budget = int(budget_match.group(1)) if budget_match else None

        def score_product(product) -> int:
            score = 0
            text_fields = " ".join(
                [
                    product["name"],
                    product["category"],
                    " ".join(product["style"]),
                    " ".join(product["season"]),
                    " ".join(product["colors"]),
                ]
            ).lower()

            for kw in ["春", "夏", "秋", "冬", "通勤", "休闲", "约会", "复古", "简约", "黑色", "蓝"]:
                if kw in q and kw in text_fields:
                    score += 2
            if budget is not None and product["price"] <= budget:
                score += 2
            if "推荐" in q or "搭配" in q:
                score += 1
            return score

        ranked = sorted(PRODUCTS, key=score_product, reverse=True)
        top_items = [item for item in ranked if score_product(item) > 0][:3]
        if not top_items:
            top_items = ranked[:3]

        lines = ["商品推荐："]
        for idx, item in enumerate(top_items, start=1):
            lines.append(
                f"{idx}. {item['name']}({item['sku']}) | 价格: {item['price']}元 | 颜色: {'/'.join(item['colors'])} | 风格: {'/'.join(item['style'])}"
            )
        lines.append("搭配建议：上紧下松或同色系叠穿更容易显瘦和提升高级感。")
        return "\n".join(lines)

    @staticmethod
    def policy_knowledge() -> str:
        return (
            "售后政策参考：\n"
            "1) 未洗涤未穿着且吊牌完整支持7天无理由退换；\n"
            "2) 质量问题支持包邮退换；\n"
            "3) 定制类商品不支持无理由退换。"
        )

    def build_business_context(self, query: str) -> str:
        intents = self.detect_intent(query)
        if not intents:
            return "无额外业务上下文"

        blocks = []
        if "size" in intents:
            blocks.append(self.recommend_size(query))
        if "style" in intents:
            blocks.append(self.recommend_products(query))
        if "policy" in intents:
            blocks.append(self.policy_knowledge())

        return "\n\n".join(blocks)
