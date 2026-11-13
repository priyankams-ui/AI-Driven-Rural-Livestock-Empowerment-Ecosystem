"""
LivestockIQ - Simple Livestock Food and Water Management Chatbot
================================================================
Provides short, practical, easy-to-understand feeding and water advice
for cattle and buffaloes in Karnataka.

Answers ONLY 8 core practical questions:
1. How much food should I give my cow/buffalo per day?
2. What food should I give in the morning?
3. What food should I give in the afternoon?
4. What food should I give in the evening?
5. Which foods should I avoid giving my livestock?
6. How much water does my cow/buffalo need per day?
7. What food and water management can support better milk production?
8. What food helps maintain the general health of my livestock?
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The 8 standard simple questions for quick chips
SIMPLE_QUESTION_CHIPS = [
    "How much food per day?",
    "What food in the morning?",
    "What food in the afternoon?",
    "What food in the evening?",
    "Which foods should I avoid?",
    "How much water per day?",
    "Better milk production",
    "Food for general health"
]

STANDARD_DISCLAIMER = "Tip: For exact individual rations or health concerns, consult a qualified veterinarian or livestock nutritionist."


class LivestockNutritionAdvisor:
    """Simple, practical food and water management assistant for livestock."""

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_or_init_session(self, session_id: str) -> Dict[str, Any]:
        """Track conversation state per session."""
        if not session_id:
            session_id = "default_session"
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "species": "Cow",
                "breed": "Jersey cattle",
                "gender": "Female",
                "age_years": 4.0,
                "milk_production": 10.0,
                "awaiting_field": None
            }
        return self.sessions[session_id]

    def parse_user_intent(self, text: str) -> str:
        """Classify user query into one of the 8 simple food & water topics."""
        t = text.lower().strip()

        # 5. Which foods should I avoid?
        if any(w in t for w in ["avoid", "toxic", "poison", "harmful", "bad food", "cannot eat", "not feed", "moldy", "spoil"]):
            return "foods_to_avoid"

        # 7. What food and water management can support better milk production?
        if any(w in t for w in ["more milk", "increase milk", "better milk", "milk production", "milk yield", "improve milk", "support milk", "milk support"]):
            return "milk_production"

        # 6. How much water does my cow/buffalo need per day?
        if any(w in t for w in ["water", "drink", "drinking", "thirsty", "liters of water", "how much water", "summer water"]):
            return "water_requirements"

        # 2. What food in the morning?
        if any(w in t for w in ["morning", "breakfast", "am feed", "start of day"]):
            return "morning_food"

        # 3. What food in the afternoon?
        if any(w in t for w in ["afternoon", "midday", "lunch", "noon"]):
            return "afternoon_food"

        # 4. What food in the evening?
        if any(w in t for w in ["evening", "night", "dinner", "pm feed"]):
            return "evening_food"

        # 8. What food helps maintain general health?
        if any(w in t for w in ["general health", "maintain health", "healthy", "immunity", "fitness", "good health", "body health"]):
            return "general_health"

        # 1. How much food should I give my cow/buffalo per day?
        if any(w in t for w in ["how much food", "how much feed", "daily food", "daily feed", "daily ration", "how many kg", "quantity", "ration", "schedule"]):
            return "daily_food_quantity"

        # Default fallback to daily food recommendation
        return "daily_food_quantity"

    def extract_context_from_query(self, text: str, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic animal parameters from natural language message."""
        ctx = dict(current_context or {})
        t = text.lower().strip()

        # Species
        if "buffalo" in t or "murrah" in t or "emme" in t:
            ctx["species"] = "Buffalo"
            if not ctx.get("breed") or "cow" in str(ctx.get("breed", "")).lower():
                ctx["breed"] = "Murrah"
        elif any(w in t for w in ["cow", "cattle", "hasu", "jersey", "holstein", "hf"]):
            ctx["species"] = "Cow"

        # Gender
        if any(w in t for w in ["bull", "ox", "bullock", "sire", "he-buffalo"]) or re.search(r'\bmale\b', t):
            ctx["gender"] = "Male"
            ctx["milk_production"] = 0.0
        elif any(w in t for w in ["cow", "female", "she-buffalo", "heifer"]):
            ctx["gender"] = "Female"

        # Breed keywords
        if "jersey" in t:
            ctx["species"] = "Cow"
            ctx["breed"] = "Jersey cattle"
        elif "holstein" in t or "hf" in t:
            ctx["species"] = "Cow"
            ctx["breed"] = "Holstein Friesian cattle"
        elif "hallikar" in t:
            ctx["species"] = "Cow"
            ctx["breed"] = "Hallikar"
        elif "murrah" in t:
            ctx["species"] = "Buffalo"
            ctx["breed"] = "Murrah"
        elif "jaffarabadi" in t:
            ctx["species"] = "Buffalo"
            ctx["breed"] = "Jaffarabadi"

        # Milk production parsing (e.g. "10 litres", "8 ltr", "yields 12L", lone "10")
        milk_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:liters?|litres?|ltrs?|ltr|l\/day|\bl\b)', t)
        if milk_match and ctx.get("gender") != "Male":
            try:
                val = float(milk_match.group(1))
                if 0.0 <= val <= 50.0:
                    ctx["milk_production"] = val
            except ValueError:
                pass
        elif re.match(r'^\s*(\d+(?:\.\d+)?)\s*$', t) and ctx.get("gender") != "Male":
            try:
                val = float(t.strip())
                if 1.0 <= val <= 50.0:
                    ctx["milk_production"] = val
            except ValueError:
                pass

        # Age parsing (e.g. "4 years old", "3 yrs")
        age_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?|yr)\s*(?:old)?', t)
        if age_match:
            try:
                val = float(age_match.group(1))
                if 0.1 <= val <= 25.0:
                    ctx["age_years"] = val
            except ValueError:
                pass

        return ctx

    # --------------------------------------------------------------------------
    # Simple Response Builders for the 8 Core Questions
    # --------------------------------------------------------------------------

    def build_daily_food_response(self, ctx: Dict[str, Any]) -> str:
        """Question 1: How much food should I give per day?"""
        species = ctx.get("species") or "Cow"
        breed = ctx.get("breed") or ("Jersey cattle" if species == "Cow" else "Murrah")
        gender = ctx.get("gender") or "Female"
        milk = float(ctx.get("milk_production") or (10.0 if gender == "Female" else 0.0))
        is_buffalo = (species == "Buffalo")

        if gender == "Male":
            conc_kg = "2.0 to 2.5 kg"
            conc_text = f"{conc_kg} of balanced concentrate feed per day for body maintenance and work stamina."
            green_kg = "20 to 25 kg"
            dry_kg = "5 to 6 kg"
        else:
            conc_rate = 500 if is_buffalo else 400
            maint_base = 2.0 if is_buffalo else 1.5
            calc_conc = round(maint_base + (milk * (0.50 if is_buffalo else 0.40)), 1)
            conc_kg = f"~{calc_conc} kg"
            conc_text = (
                f"**{conc_kg} of balanced concentrate feed per day** "
                f"({maint_base} kg for body maintenance + {conc_rate}g for each litre of milk produced). "
                f"Divide this into two equal morning and evening portions."
            )
            green_kg = "25 to 30 kg" if is_buffalo else "20 to 25 kg"
            dry_kg = "5 to 6 kg" if is_buffalo else "4 to 5 kg"

        lines = [
            f"### 🌾 Daily Food Guide for {breed} ({species})",
            f"*Profile: {gender} | Milk Yield: {milk} Litres/day*" if gender == "Female" else f"*Profile: Male ({breed} {species})*",
            "",
            "Here is the simple, practical daily food requirement:",
            "",
            f"- **Green Fodder:** **{green_kg}** of fresh chopped green grass (such as Super Napier, fodder maize, or jowar).",
            f"- **Dry Fodder:** **{dry_kg}** of good dry straw (such as Ragi straw or Paddy straw).",
            f"- **Concentrate Feed:** {conc_text}",
            f"- **Mineral Mixture & Salt:** **50g** area-specific mineral mixture and **30g** common salt daily.",
            "",
            "> **Important — Fresh Fodder vs Dry Matter:**",
            "> Fresh green grass contains **75% to 85% water**, meaning 20–25 kg of fresh green fodder provides only about 4–5 kg of solid dry nutrients. Dry straw provides the essential fiber needed for healthy rumination and digestion. Never replace dry straw completely with wet green grass."
        ]
        return "\n".join(lines)

    def build_morning_food_response(self, ctx: Dict[str, Any]) -> str:
        """Question 2: What food should I give in the morning?"""
        species = ctx.get("species") or "Cow"
        breed = ctx.get("breed") or ("Jersey cattle" if species == "Cow" else "Murrah")

        lines = [
            f"### 🌅 Morning Feeding for {breed} ({species})",
            "",
            "Give the following simple morning feeding:",
            "",
            "- **Clean drinking water:** Offer fresh, clean water first thing in the morning.",
            "- **Concentrate feed:** Provide half of the daily concentrate portion before or during morning milking. Mix in half of the daily mineral mixture.",
            "- **Green fodder:** Offer fresh chopped green grass (such as Napier or green maize) after milking.",
            "- **Dry fodder:** Give a small portion (1 to 2 kg) of dry ragi or paddy straw to stimulate saliva flow and digestion.",
            "",
            "> **Tip:** Always chop green and dry fodder to 2–3 cm pieces to reduce waste and make chewing easier."
        ]
        return "\n".join(lines)

    def build_afternoon_food_response(self, ctx: Dict[str, Any]) -> str:
        """Question 3: What food should I give in the afternoon?"""
        species = ctx.get("species") or "Cow"
        breed = ctx.get("breed") or ("Jersey cattle" if species == "Cow" else "Murrah")
        is_buffalo = (species == "Buffalo")

        buffalo_cooling = (
            "- **Water splashing for buffaloes:** Buffaloes have dark skin and very few sweat glands. "
            "Splash cool water over your buffalo's body during hot afternoon hours (11:00 AM – 3:00 PM) to prevent heat stress."
        ) if is_buffalo else (
            "- **Cool shelter:** Keep cows in a well-ventilated, shaded shed away from direct midday sun."
        )

        lines = [
            f"### ☀️ Afternoon Feeding & Care for {breed} ({species})",
            "",
            "In the afternoon, follow these simple guidelines:",
            "",
            "- **Dry fodder:** Provide good-quality dry straw (Ragi straw or Paddy straw). Chewing dry straw at midday keeps the stomach active and aids proper digestion.",
            "- **Fresh drinking water:** Ensure the water trough is full of clean, cool drinking water so the animal can drink freely.",
            buffalo_cooling,
            "",
            "> **Tip:** Avoid feeding heavy grain or concentrate during hot midday hours."
        ]
        return "\n".join(lines)

    def build_evening_food_response(self, ctx: Dict[str, Any]) -> str:
        """Question 4: What food should I give in the evening?"""
        species = ctx.get("species") or "Cow"
        breed = ctx.get("breed") or ("Jersey cattle" if species == "Cow" else "Murrah")

        lines = [
            f"### 🌇 Evening Feeding for {breed} ({species})",
            "",
            "Give the following simple evening feeding:",
            "",
            "- **Clean drinking water:** Allow the animal to drink clean water before evening milking.",
            "- **Concentrate feed:** Provide the second half of the daily concentrate portion before or during evening milking. Add common salt (about 30g).",
            "- **Green fodder:** Feed the remaining portion of fresh chopped green grass.",
            "- **Dry roughage for the night:** Leave 2 to 3 kg of dry straw in the manger for the animal to chew calmly overnight.",
            "",
            "> **Tip:** Keep clean water accessible in the shed throughout the night."
        ]
        return "\n".join(lines)

    def build_foods_to_avoid_response(self, ctx: Dict[str, Any]) -> str:
        """Question 5: Which foods should I avoid giving my livestock?"""
        species = ctx.get("species") or "Cow"

        lines = [
            f"### ⚠️ Foods to Avoid Giving Your {species}",
            "",
            "For your animal's safety, strictly avoid giving:",
            "",
            "- **Moldy or fungus-infected fodder:** Never feed moldy grain or straw. Mold produces harmful toxins (aflatoxins) that damage the liver and reduce milk.",
            "- **Rotten food or spoiled kitchen waste:** Stale kitchen waste or spoiled food can carry dangerous bacteria and cause severe food poisoning.",
            "- **Young or stunted jowar/sorghum (< 45 days old):** Immature jowar contains toxic prussic acid (HCN) that can be fatal to cattle and buffaloes.",
            "- **Poisonous pasture weeds:** Keep animals away from toxic plants like *Lantana* (unni gida), *Datura*, and *Parthenium*.",
            "- **Excessive concentrate or grain at once:** Feeding too much grain, flour, or broken rice at one time causes severe stomach acidity (acidosis) and life-threatening bloat.",
            "- **Dirty or contaminated water:** Stagnant, dirty water spreads parasites and bacterial infections.",
            "",
            "> **Important:** Always check feeds for bad smell, dampness, or discoloration before feeding."
        ]
        return "\n".join(lines)

    def build_water_response(self, ctx: Dict[str, Any]) -> str:
        """Question 6: How much water does my cow/buffalo need per day?"""
        species = ctx.get("species") or "Cow"
        breed = ctx.get("breed") or ("Jersey cattle" if species == "Cow" else "Murrah")
        gender = ctx.get("gender") or "Female"
        milk = float(ctx.get("milk_production") or 0.0) if gender == "Female" else 0.0
        is_buffalo = (species == "Buffalo")

        base_litres = "70 to 100+" if is_buffalo else "50 to 80"
        milk_multiplier = "4.5" if is_buffalo else "3.5"

        lines = [
            f"### 💧 Water Requirements for {breed} ({species})",
            "",
            "Water is the most important daily requirement for your livestock:",
            "",
            f"- **Daily Requirement:** An adult {species.lower()} typically requires **{base_litres} litres** of clean water every day.",
            f"- **Lactating Animals:** Require an additional **{milk_multiplier} litres of water for every 1 litre of milk produced**.",
            "- **Summer Weather:** Water intake increases by **30% to 50%** during hot summer months.",
            "- **Continuous Availability:** Clean, fresh drinking water should **remain freely available at all times throughout the day (24/7)**. Never restrict water to only 2 or 3 times a day.",
            "- **Post-Milking Thirst:** Cows and buffaloes drink nearly 40% of their daily water right after milking, so keep troughs filled.",
            "",
            "> **Signs of lack of water:** Sudden drop in daily milk yield, dry muzzle, hard dung balls, and dull eyes."
        ]
        return "\n".join(lines)

    def build_milk_production_response(self, ctx: Dict[str, Any]) -> str:
        """Question 7: What food and water management can support better milk production?"""
        species = ctx.get("species") or "Cow"
        breed = ctx.get("breed") or ("Jersey cattle" if species == "Cow" else "Murrah")
        is_buffalo = (species == "Buffalo")
        conc_ratio = "500g" if is_buffalo else "400g"

        lines = [
            f"### 🥛 Food & Water Guide for Better Milk Production: {breed} ({species})",
            "",
            "To support healthy, sustained milk production, follow these practical steps:",
            "",
            "- **Balanced fodder mix:** Combine cereal green grass (Napier or maize) with protein-rich legume green (Lucerne or cowpea) and good dry straw.",
            f"- **Adequate concentrate feed:** Provide balanced feed according to milk yield ({conc_ratio} of concentrate per liter of milk produced).",
            "- **Daily mineral mixture & salt:** Add 50–60g of area-specific mineral mixture and 30g salt to daily feed to keep the udder and body healthy.",
            "- **Continuous clean water:** 87% of milk is water. A shortage of clean water immediately causes a drop in milk volume.",
            "- **Consistent feeding and milking time:** Always feed and milk at the exact same hours every morning and evening.",
            "- **Comfortable, cool shelter:** Protect animals from heat stress and keep bedding clean and dry to prevent mastitis.",
            "",
            "> **Important:** Milk yield is fundamentally limited by the animal's breed, genetics, and health. No single feed can magically increase milk by a set number of litres. Never overfeed grain, as it causes stomach acidosis."
        ]
        return "\n".join(lines)

    def build_general_health_response(self, ctx: Dict[str, Any]) -> str:
        """Question 8: What food helps maintain general health?"""
        species = ctx.get("species") or "Cow"
        breed = ctx.get("breed") or ("Jersey cattle" if species == "Cow" else "Murrah")

        lines = [
            f"### 🌿 Food for General Health Maintenance: {breed} ({species})",
            "",
            "To keep your livestock healthy, active, and disease-free:",
            "",
            "- **Three-part balanced diet:** Ensure your animal gets green grass, dry straw, and a small portion of balanced concentrate feed every day.",
            "- **Dry straw is mandatory:** Dry ragi or paddy straw keeps the rumen chewing and prevents loose dung or stomach upsets.",
            "- **Daily mineral mixture:** A daily handful (50g) of mineral mixture prevents mineral deficiencies and strengthens the immune system.",
            "- **Common salt:** 30g of salt daily encourages adequate drinking and maintains proper electrolyte balance.",
            "- **Continuous clean drinking water:** Free access to fresh, cool drinking water 24 hours a day.",
            "- **Gradual feed changes:** When introducing any new feed or grain, mix it slowly over 7 to 10 days to let the stomach adjust.",
            "",
            "> **Tip:** Good nutrition, regular deworming, and clean shelter are the foundations of strong animal health."
        ]
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # Master Processing Entrypoint
    # --------------------------------------------------------------------------
    def generate_nutrition_advice(
        self,
        user_message: str,
        animal_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process incoming chat query and return simple, practical answer."""
        session_id = session_id or "default_session"
        session_state = self.get_or_init_session(session_id)

        # Merge animal context from request
        incoming_ctx = dict(animal_context or {})
        for k, v in incoming_ctx.items():
            if v is not None and v != "":
                session_state[k] = v

        # Parse message entities
        session_state = self.extract_context_from_query(user_message, session_state)
        intent = self.parse_user_intent(user_message)

        logger.info(f"[Simple Nutrition Advisor] Session: {session_id} | Intent: {intent} | Species: {session_state.get('species')}")

        # Generate response strictly for the 8 questions
        if intent == "morning_food":
            reply_text = self.build_morning_food_response(session_state)
        elif intent == "afternoon_food":
            reply_text = self.build_afternoon_food_response(session_state)
        elif intent == "evening_food":
            reply_text = self.build_evening_food_response(session_state)
        elif intent == "foods_to_avoid":
            reply_text = self.build_foods_to_avoid_response(session_state)
        elif intent == "water_requirements":
            reply_text = self.build_water_response(session_state)
        elif intent == "milk_production":
            reply_text = self.build_milk_production_response(session_state)
        elif intent == "general_health":
            reply_text = self.build_general_health_response(session_state)
        else:
            # Question 1: daily food quantity
            reply_text = self.build_daily_food_response(session_state)

        return {
            "status": "success",
            "reply": reply_text,
            "intent": intent,
            "animal_context": session_state,
            "suggested_questions": SIMPLE_QUESTION_CHIPS,
            "disclaimer": STANDARD_DISCLAIMER
        }


# Singleton instance
nutrition_advisor = LivestockNutritionAdvisor()
