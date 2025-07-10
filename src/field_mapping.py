import os
import re
import json
from src import config
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class TaxDocumentExtractor:
    def __init__(self):
        self.groq_client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )

        self.field_descriptions = {
            # Basic Info
            "taxpayer_name": "Extract the primary taxpayer's full legal name as written on the tax form",
            "spouse_name": "Extract the spouse's full legal name if filing jointly, otherwise leave empty",
            "ssn": "Extract the primary taxpayer's social security number (format: XXX-XX-XXXX)",
            "spouse_ssn": "Extract the spouse's social security number if filing jointly (format: XXX-XX-XXXX)",
            "address": "Extract the complete mailing address including street number, street name, and unit/apt if applicable",
            "city": "Extract the city name from the mailing address",
            "state": "Extract the two-letter state abbreviation from the mailing address",
            "zip_code": "Extract the ZIP code from the mailing address (5-digit or 9-digit format)",
            "filing_status": "Identify the filing status from checked boxes: Single, Married Filing Jointly, Married Filing Separately, Head of Household, or Qualifying Widow(er)",

            # Income Fields (Form 1040)
            "wages": "Extract the dollar amount of total wages, salaries, tips from W-2 forms (ignore line numbers, extract only the monetary value)",
            "taxable_interest": "Extract the dollar amount of taxable interest income from banks, credit unions, etc. (ignore line numbers, extract only the monetary value)",
            "qualified_dividends": "Extract the dollar amount of qualified dividends that receive preferential tax treatment (ignore line numbers, extract only the monetary value)",
            "ordinary_dividends": "Extract the dollar amount of total ordinary dividends including both qualified and non-qualified (ignore line numbers, extract only the monetary value)",
            "capital_gains_or_loss": "Extract the dollar amount of net capital gain or loss from investment sales (ignore line numbers, extract only the monetary value)",
            "total_income": "Extract the dollar amount of the sum of all income sources before adjustments (ignore line numbers, extract only the monetary value)",
            "adjusted_gross_income": "Extract the dollar amount of AGI after subtracting above-the-line deductions from total income (ignore line numbers, extract only the monetary value)",
            "standard_or_itemized_deduction": "Extract the dollar amount of the deduction used (either standard deduction or total itemized) (ignore line numbers, extract only the monetary value)",
            "total_deductions": "Extract the dollar amount of the total deduction claimed (ignore line numbers, extract only the monetary value)",
            "taxable_income": "Extract the dollar amount of the final taxable income after subtracting deductions from AGI (ignore line numbers, extract only the monetary value)",

            # Tax Section (Form 1040)
            "income_tax": "Extract the calculated income tax before credits",
            "child_tax_credit": "Extract the child tax credit amount claimed",
            "other_credits": "Extract total of all other tax credits claimed",
            "total_credits": "Extract the sum of child tax credit and other credits",
            "additional_taxes": "Extract additional taxes including self-employment tax, alternative minimum tax, etc.",
            "total_tax": "Extract the final total tax liability after credits and additional taxes",

            # Payment Section (Form 1040)
            "federal_withholding_w2": "Extract federal income tax withheld from W-2 forms",
            "federal_withholding_1099": "Extract federal income tax withheld from 1099 forms",
            "other_withholding": "Extract any other federal income tax withheld",
            "federal_total_withholding": "Extract total federal income tax withheld from all sources",
            "estimated_payments": "Extract estimated tax payments made during the year",
            "total_payments": "Extract total tax payments and credits",
            "refund": "Extract refund amount if total payments exceed total tax",
            "amount_owed": "Extract amount owed if total tax exceeds total payments",

            # Schedule 1 - Additional Income and Adjustments
            "additional_income": "Extract total additional income from Schedule 1",
            "adjustments_to_income": "Extract total adjustments to income from Schedule 1",
            "taxable_refunds": "Extract taxable state and local tax refunds",
            "alimony_received": "Extract alimony received under pre-2019 divorce agreements",
            "business_income": "Extract business income or loss from Schedule C",
            "capital_gain_loss": "Extract capital gain or loss from Schedule D",
            "other_gains_losses": "Extract other gains or losses from Form 4797",
            "rental_income": "Extract rental real estate income from Schedule E",
            "unemployment_compensation": "Extract unemployment compensation received",
            "other_income": "Extract other miscellaneous income reported",
            "educator_expenses": "Extract qualified educator expenses deduction",
            "business_expenses": "Extract business expenses for reservists, performing artists, etc.",
            "hsa_deduction": "Extract health savings account deduction",
            "moving_expenses": "Extract moving expenses for military members",
            "se_tax_deduction": "Extract deductible portion of self-employment tax",
            "sep_simple_ira": "Extract SEP, SIMPLE, and qualified retirement plan contributions",
            "self_employed_health": "Extract self-employed health insurance premiums",
            "penalty_early_withdrawal": "Extract penalty for early withdrawal of savings",
            "alimony_paid": "Extract alimony paid under pre-2019 divorce agreements",
            "ira_deduction": "Extract IRA contribution deduction",
            "student_loan_interest": "Extract student loan interest deduction",

            # Schedule 2 - Additional Taxes
            "amt": "Extract alternative minimum tax amount",
            "excess_advance_ptc": "Extract excess advance premium tax credit repayment",
            "additional_taxes_other": "Extract other additional taxes",
            "total_schedule_2": "Extract total additional taxes from Schedule 2",

            # Schedule 3 - Additional Credits and Payments
            "foreign_tax_credit": "Extract the dollar amount for foreign tax credit (ignore line numbers, extract only the monetary value)",
            "child_dependent_care_credit": "Extract the dollar amount for child and dependent care credit (ignore line numbers, extract only the monetary value)",
            "education_credits": "Extract the dollar amount for education credits (ignore line numbers, extract only the monetary value)",
            "retirement_savings_credit": "Extract the dollar amount for retirement savings contributions credit (ignore line numbers, extract only the monetary value)",
            "residential_energy_credit": "Extract the dollar amount for residential clean energy credit (ignore line numbers, extract only the monetary value)",
            "other_nonrefundable_credits": "Extract the dollar amount for other nonrefundable credits (ignore line numbers, extract only the monetary value)",
            "total_other_credits": "Extract the total dollar amount of other credits (ignore line numbers, extract only the monetary value)",
            "net_ptc": "Extract the dollar amount for net premium tax credit (ignore line numbers, extract only the monetary value)",
            "amount_paid_extension": "Extract the dollar amount paid with extension request (ignore line numbers, extract only the monetary value)",
            "excess_ss_tax": "Extract the dollar amount of excess social security tax withheld (ignore line numbers, extract only the monetary value)",
            "credit_tax_paid_forms": "Extract the dollar amount for credit for tax paid on undistributed capital gains (ignore line numbers, extract only the monetary value)",
            "other_payments": "Extract the dollar amount for other payments and refundable credits (ignore line numbers, extract only the monetary value)",
            "total_other_payments": "Extract the total dollar amount of other payments (ignore line numbers, extract only the monetary value)",

            # Schedule A - Itemized Deductions
            "medical_dental": "Extract medical and dental expenses exceeding AGI threshold",
            "state_local_income_tax": "Extract state and local income taxes paid",
            "state_local_sales_tax": "Extract state and local general sales taxes (if elected instead of income tax)",
            "real_estate_tax": "Extract real estate taxes paid",
            "personal_property_tax": "Extract personal property taxes paid",
            "other_taxes": "Extract other deductible taxes",
            "home_mortgage_interest": "Extract home mortgage interest on acquisition debt",
            "home_mortgage_points": "Extract home mortgage points paid",
            "mortgage_insurance_premiums": "Extract mortgage insurance premiums",
            "investment_interest": "Extract investment interest expense",
            "charitable_cash": "Extract charitable contributions by cash or check",
            "charitable_noncash": "Extract charitable contributions other than cash",
            "charitable_carryover": "Extract charitable contribution carryover from prior years",
            "casualty_theft_losses": "Extract casualty and theft losses from federally declared disasters",
            "other_itemized": "Extract other miscellaneous itemized deductions",
            "total_itemized": "Extract total itemized deductions from Schedule A",

            # Schedule B - Interest and Ordinary Dividends
            "total_interest": "Extract total taxable interest income",
            "total_dividends": "Extract total ordinary dividends received",
            "foreign_accounts": "Extract answer to foreign financial accounts question (Yes/No)",
            "foreign_trust": "Extract answer to foreign trusts question (Yes/No)",

            # Schedule C - Profit or Loss from Business
            "gross_receipts": "Extract gross receipts or sales from business",
            "returns_allowances": "Extract returns and allowances",
            "cost_of_goods_sold": "Extract cost of goods sold",
            "gross_profit": "Extract gross profit from business",
            "total_expenses": "Extract total business expenses",
            "net_profit_loss": "Extract net profit or loss from business",

            # Schedule D - Capital Gains and Losses
            "short_term_gain_loss": "Extract net short-term capital gain or loss",
            "long_term_gain_loss": "Extract net long-term capital gain or loss",
            "total_capital_gain_loss": "Extract total capital gain or loss",
            "capital_gain_distributions": "Extract capital gain distributions from mutual funds",
            "unrecaptured_section_1250": "Extract unrecaptured section 1250 gain",
            "collectibles_gain": "Extract 28% rate gain from collectibles",

            # Schedule E - Supplemental Income and Loss
            "rental_real_estate_income": "Extract total rental real estate and royalty income or loss",
            "royalty_income": "Extract royalty income",
            "partnership_s_corp_income": "Extract partnership and S corporation income or loss",
            "estate_trust_income": "Extract estate and trust income or loss",
            "total_supplemental_income": "Extract total supplemental income from Schedule E",

            # Schedule F - Profit or Loss from Farming
            "gross_farm_income": "Extract gross farm income",
            "farm_expenses": "Extract total farm expenses",
            "net_farm_profit_loss": "Extract net farm profit or loss",

            # Schedule H - Household Employment Taxes
            "household_wages": "Extract total cash wages paid to household employees",
            "household_ss_medicare": "Extract social security and Medicare taxes on household wages",
            "household_futa": "Extract federal unemployment tax on household wages",
            "total_household_tax": "Extract total household employment taxes",

            # Schedule J - Income Averaging for Farmers and Fishermen
            "elected_farm_income": "Extract elected farm income for averaging",
            "average_income": "Extract average income using income averaging",

            # Schedule R - Credit for the Elderly or Disabled
            "credit_elderly_disabled": "Extract credit for the elderly or disabled",

            # Schedule SE - Self-Employment Tax
            "net_earnings_se": "Extract net earnings from self-employment",
            "self_employment_tax": "Extract total self-employment tax",
            "deductible_se_tax": "Extract deductible portion of self-employment tax",

            # Form 4868 - Extension
            "extension_payment": "Extract amount paid with automatic extension request",

            # Form 8812 - Additional Child Tax Credit
            "additional_child_tax_credit": "Extract additional child tax credit (refundable portion)",
            "earned_income": "Extract earned income for child tax credit calculation",

            # Form 8829 - Expenses for Business Use of Home
            "home_office_area": "Extract area of home used for business (square feet)",
            "total_home_area": "Extract total area of home (square feet)",
            "home_office_percentage": "Extract percentage of home used for business",
            "home_office_deduction": "Extract allowable home office deduction",

            # Schedule EIC - Earned Income Credit
            "earned_income_credit": "Extract earned income credit amount",
            "qualifying_children_eic": "Extract number of qualifying children for EIC"
        }


    def extract_section_data_llm(self, section_text: str, section_name: str, field_list: List[str]) -> Dict[str, Any]:
        data = {}
        print(f"🔍 Parsing section via LLM: {section_name}")
        if not self.groq_client:
            raise ValueError("Groq client not initialized. Set GROQ_API_KEY.")

        fields_description = "\n".join(
            [f"- {field}: {self.field_descriptions.get(field, field)}" for field in field_list]
        )
        
        prompt = f"""
        You are an IRS Form 1040 parsing expert.

        INSTRUCTIONS:
        - Extract the requested fields listed below. Return a **valid JSON only**, no commentary.
        - If a field is **missing entirely**, set its value to `null`.
        - If a field is **present but blank or shows $0**, return 0.
        - For **all monetary fields**, extract ONLY the actual dollar amount **entered by the taxpayer** — do NOT return labels or line numbers.

        ⚠️ IMPORTANT RULES FOR TABLE FORMATTING:
        - **Do NOT extract line numbers (e.g., 1, 2, 3)** — these are structural references, NOT taxpayer data.
        - When parsing a table:
        1. For each field, locate the field name or keyword.
        2. Extract the **numeric value that appears to the right of the final pipe (`|`) on that line**, if it exists and is not a line number.
            - ✅ Example: `"1 Foreign tax credit ... |  |  | 1 | 5"` → `"foreign_tax_credit": 5`
            - ❌ Do NOT return `1` as the value, that’s the line number, not data.
            - ❌ If a field appears but there is no value after the last `|`, return `0` or `null`.

        FIELDS TO EXTRACT:
        {fields_description}

        SECTION START
        {section_text}
        SECTION END

        Return JSON like:
        {{
        "field_1": value,
        "field_2": value,
        ...
        }}
        """
        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=800
            )
            # print("raw_ouput",response.choices[0].message.content)
            raw_output = response.choices[0].message.content.strip()
            
            # Try extracting JSON block
            json_block_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if not json_block_match:
                raise ValueError("No JSON block found in LLM response.")
            
            parsed = json.loads(json_block_match.group(0))
            return {field: parsed.get(field, 0) for field in field_list}
        
        except Exception as e:
            print(f"⚠️ LLM extraction error in {section_name}: {e}")
            return {field: 0 for field in field_list}

    def extract_all_sections(self,text: str):
        def extract_by_headers(start: str, end_pattern: str) -> str:
            end_matches = re.findall(end_pattern, text)
            if start in text and end_matches:
                end_text = end_matches[-1]
                return text[text.find(start):text.rfind(end_text) + len(end_text)]
            return ""

        year_pattern_schedules = r"\(Form 1040\)\s*\d{4}"
        year_pattern_forms = r"\(\d{4}\)"

        sections = {
            "basic_info": re.search(r'(?s)Your first name and middle initial .*?(?=1a\s+)', text)
                            or re.search(r'(?s)U.S. Individual Income Tax Return.*?(?=1a\s+)', text),
            "income": re.search(r'(?s)TABLE:\sIncome \| 1a \| Total amount from Form.*?Form 1040', text)
                      or re.search(r'(?s)(?:1[az]|W-2).*?15\s+[\d,\-]+', text),
            "tax": re.search(r'(?s)TABLE:\sTax and\s*\|.*?total tax[\|\s\d\,]+', text)
                    or re.search(r'(?s)Tax and\s+16.*?24\s+[\d,\-]+', text),
            "payments": re.search(r'(?s)(?:Payments\s*\|)?\s*25\s*\|.*?37\s*\|.*?(?=38\s*\|)', text)
                         or re.search(r'(?s)Payments\s+25.*?37\s+[\d,\-]+', text),

            # Schedules and forms (with fallback)
            "schedule_a": extract_by_headers("SCHEDULE A", f"Schedule A {year_pattern_schedules}"),
            "schedule_b": extract_by_headers("SCHEDULE B", f"Schedule B {year_pattern_schedules}"),
            "schedule_c": extract_by_headers("SCHEDULE C", f"Schedule C {year_pattern_schedules}"),
            "schedule_d": extract_by_headers("SCHEDULE D", f"Schedule D {year_pattern_schedules}"),
            "schedule_e": extract_by_headers("SCHEDULE E", f"Schedule E {year_pattern_schedules}"),
            "schedule_f": extract_by_headers("SCHEDULE F", f"Schedule F {year_pattern_schedules}"),
            "schedule_h": extract_by_headers("SCHEDULE H", f"Schedule H {year_pattern_schedules}"),
            "schedule_j": extract_by_headers("SCHEDULE J", f"Schedule J {year_pattern_schedules}"),
            "schedule_r": extract_by_headers("SCHEDULE R", f"Schedule R {year_pattern_schedules}"),
            "schedule_se": extract_by_headers("SCHEDULE SE", f"Schedule SE {year_pattern_schedules}"),
            "schedule_1": extract_by_headers("SCHEDULE 1", f"Schedule 1 {year_pattern_schedules}"),
            "schedule_2": extract_by_headers("SCHEDULE 2", f"Schedule 2 {year_pattern_schedules}"),
            "schedule_3": extract_by_headers("SCHEDULE 3", f"Schedule 3 {year_pattern_schedules}"),

            "form_8949": extract_by_headers("Form Sales and Other Dispositions of Capital Assets", r"Form 8949 \(\d{4}\)|Form: 8949 \(\d{4}\)"),
            "form_4868": extract_by_headers("Form 4868", f"Form 4868 {year_pattern_forms}"),
            "form_8812": extract_by_headers("Form 8812", f"Form 8812 {year_pattern_forms}"),
            "form_8829": extract_by_headers("Form 8829", f"Form 8829 {year_pattern_forms}"),
            "schedule_eic": extract_by_headers("Schedule EIC", f"Schedule EIC {year_pattern_schedules}"),
            "sign_here": extract_by_headers("Sign Under penalties of perjury", "Spouse's occupation"),
            "paid_preparer": extract_by_headers("Preparer's signature", "Firm's EIN")
        }
        # print("basic_info", sections["basic_info"].group(0) if sections["basic_info"] else '')
        sections["basic_info"]=sections["basic_info"].group(0) if sections["basic_info"] else ''
        sections["income"]=sections["income"].group(0) if sections["income"] else ''
        sections['tax']=sections["tax"].group(0) if sections["tax"] else ''
        sections["payments"]=sections["payments"].group(0) if sections["payments"] else ''
        extracted_data = {
            "form_type": "Form 1040",
            "tax_year":  re.search(r"Form 1040 Department of the Treasury-Internal Revenue Service (\d{4})", text).group(1),
            "basic_info": self.extract_section_data_llm(
                sections["basic_info"],
                "basic_info",
                ["taxpayer_name", "spouse_name", "ssn", "spouse_ssn", "address", "city", "state", "zip_code", "filing_status"]
            ),
            "income_section": self.extract_section_data_llm(
                sections["income"],
                "income", [
                    "wages", "taxable_interest", "qualified_dividends", "ordinary_dividends",
                    "capital_gains_or_loss", "total_income", "adjusted_gross_income",
                    "standard_or_itemized_deduction", "total_deductions", "taxable_income"
                ]
            ),
            "tax_section": self.extract_section_data_llm(
                sections["tax"],
                "tax", [
                    "income_tax", "child_tax_credit", "other_credits", "total_credits",
                    "additional_taxes", "total_tax"
                ]
            ),
            "payment_section": self.extract_section_data_llm(
                sections["payments"],
                "payments", [
                    "federal_withholding_w2", "federal_withholding_1099", "other_withholding",
                    "federal_total_withholding", "estimated_payments", "total_payments",
                    "refund", "amount_owed"
                ]
            ),
            "schedule_1": self.extract_section_data_llm(
                sections["schedule_1"],
                "schedule_1", [
                    "additional_income", "adjustments_to_income", "taxable_refunds", "alimony_received",
                    "business_income", "capital_gain_loss", "other_gains_losses", "rental_income",
                    "unemployment_compensation", "other_income", "educator_expenses", "business_expenses",
                    "hsa_deduction", "moving_expenses", "se_tax_deduction", "sep_simple_ira",
                    "self_employed_health", "penalty_early_withdrawal", "alimony_paid", "ira_deduction",
                    "student_loan_interest"
                ]
            ),
            "schedule_2": self.extract_section_data_llm(
                sections["schedule_2"],
                "schedule_2", [
                    "amt", "excess_advance_ptc", "additional_taxes_other", "total_schedule_2"
                ]
            ),
            "schedule_3": self.extract_section_data_llm(
                sections["schedule_3"],
                "schedule_3", [
                    "foreign_tax_credit", "child_dependent_care_credit", "education_credits",
                    "retirement_savings_credit", "residential_energy_credit", "other_nonrefundable_credits",
                    "total_other_credits", "net_ptc", "amount_paid_extension", "excess_ss_tax",
                    "credit_tax_paid_forms", "other_payments", "total_other_payments"
                ]
            ),
            "schedule_a": self.extract_section_data_llm(
                sections["schedule_a"],
                "schedule_a", [
                    "medical_dental", "state_local_income_tax", "state_local_sales_tax", "real_estate_tax",
                    "personal_property_tax", "other_taxes", "home_mortgage_interest", "home_mortgage_points",
                    "mortgage_insurance_premiums", "investment_interest", "charitable_cash", "charitable_noncash",
                    "charitable_carryover", "casualty_theft_losses", "other_itemized", "total_itemized"
                ]
            ),
            "schedule_b": self.extract_section_data_llm(
                sections["schedule_b"],
                "schedule_b", [
                    "total_interest", "total_dividends", "foreign_accounts", "foreign_trust"
                ]
            ),
            "schedule_c": self.extract_section_data_llm(
                sections["schedule_c"],
                "schedule_c", [
                    "gross_receipts", "returns_allowances", "cost_of_goods_sold", "gross_profit",
                    "total_expenses", "net_profit_loss"
                ]
            ),
            "schedule_d": self.extract_section_data_llm(
                sections["schedule_d"],
                "schedule_d", [
                    "short_term_gain_loss", "long_term_gain_loss", "total_capital_gain_loss",
                    "capital_gain_distributions", "unrecaptured_section_1250", "collectibles_gain"
                ]
            ),
            "schedule_e": self.extract_section_data_llm(
                sections["schedule_e"],
                "schedule_e", [
                    "rental_real_estate_income", "royalty_income", "partnership_s_corp_income",
                    "estate_trust_income", "total_supplemental_income"
                ]
            ),
            "schedule_f": self.extract_section_data_llm(
                sections["schedule_f"],
                "schedule_f", [
                    "gross_farm_income", "farm_expenses", "net_farm_profit_loss"
                ]
            ),
            "schedule_h": self.extract_section_data_llm(
                sections["schedule_h"],
                "schedule_h", [
                    "household_wages", "household_ss_medicare", "household_futa", "total_household_tax"
                ]
            ),
            "schedule_j": self.extract_section_data_llm(
                sections["schedule_j"],
                "schedule_j", [
                    "elected_farm_income", "average_income"
                ]
            ),
            "schedule_r": self.extract_section_data_llm(
                sections["schedule_r"],
                "schedule_r", [
                    "credit_elderly_disabled"
                ]
            ),
            "schedule_se": self.extract_section_data_llm(
                sections["schedule_se"],
                "schedule_se", [
                    "net_earnings_se", "self_employment_tax", "deductible_se_tax"
                ]
            ),
            "form_4868": self.extract_section_data_llm(
                sections["form_4868"],
                "form_4868", [
                    "extension_payment"
                ]
            ),
            "form_8812": self.extract_section_data_llm(
                sections["form_8812"],
                "form_8812", [
                    "additional_child_tax_credit", "earned_income"
                ]
            ),
            "form_8829": self.extract_section_data_llm(
                sections["form_8829"],
                "form_8829", [
                    "home_office_area", "total_home_area", "home_office_percentage", "home_office_deduction"
                ]
            ),
            "schedule_eic": self.extract_section_data_llm(
                sections["schedule_eic"],
                "schedule_eic", [
                    "earned_income_credit", "qualifying_children_eic"
                ]
            )
        }
        return extracted_data,sections


def field_map(raw_text_path, session_id):
    print("file name",raw_text_path)
    with open(raw_text_path, "r", encoding="utf-8") as f:
        content = f.read()

    extractor = TaxDocumentExtractor()
    result, sectioned_data = extractor.extract_all_sections(content)

    json_name = f'{session_id}_fields.json'
    json_path = config.PATHS.get("json_data_path", "")

    save_path = os.path.join(json_path, json_name)
    save_path = config.PATHS.get("section_save_path","")
    save_name=f'{session_id}_sectioned_data.json'
    save_path=os.path.join(save_path,save_name)

    # Save proper JSON object (not stringified JSON)
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)
    print("saving secioned_data....!")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(sectioned_data, f, indent=2, ensure_ascii=False)
    

    return result,sectioned_data
    