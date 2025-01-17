"""
UAE Transfer Pricing AI Assessment Tool
---------------------------------------

Streamlit + LangChain application to assess whether two parties are Related Parties
or Connected Persons under UAE Transfer Pricing regulations. 

This version uses st.secrets for the OPENAI_API_KEY. 
Make sure to set `openai_api_key = st.secrets["openai_api_key"]` in your Streamlit
secrets to securely pass your API key on Streamlit Cloud.

Usage:
1. Deploy on Streamlit Cloud.
2. In your Streamlit secrets, add:
   [general]
   openai_api_key = "YOUR_ACTUAL_OPENAI_API_KEY"
3. Run with: streamlit run main.py
"""

import streamlit as st
from langchain.chat_models import ChatOpenAI
from typing import Any, Dict, List

# -------------------------------------------------------------------------
# 1. Model Initialization (Read key from st.secrets)
# -------------------------------------------------------------------------
def get_llama_model():
    """
    Returns an instance of the ChatOpenAI model with a Llama backend.
    The OPENAI_API_KEY is retrieved from Streamlit secrets.
    """
    model = ChatOpenAI(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        openai_api_key=st.secrets["openai_api_key"],  # From secrets
        openai_api_base="https://api.together.xyz/v1"  # Example base URL (adjust if needed)
    )
    return model

# -------------------------------------------------------------------------
# 2. Configuration Constants and Thresholds
# -------------------------------------------------------------------------
CONFIG = {
    "OWNERSHIP_THRESHOLD": 50,
    "CONTROL_THRESHOLD": 50,
    "MAX_FAMILY_DEGREE": 4,
    "DOCUMENTATION_THRESHOLDS": {
        "REVENUE": 200_000_000,
        "MNE_GROUP_CONSOLIDATED": 3_150_000_000
    }
}

# -------------------------------------------------------------------------
# 3. Family Relationship Analyzer
# -------------------------------------------------------------------------
class FamilyRelationshipAnalyzer:
    """
    Analyzes family relationships up to the 4th degree for determining 
    Related Party status among individuals.
    """
    def __init__(self):
        self.family_degrees = {
            1: ["parent", "child", "spouse_parent", "spouse_child"],
            2: ["grandparent", "grandchild", "sibling", "spouse_sibling"],
            3: ["great_grandparent", "great_grandchild", "uncle", "aunt", "niece", "nephew"],
            4: ["great_great_grandparent", "great_great_grandchild", "first_cousin"]
        }

    def assess_relationship(self, person1: Dict[str, Any], person2: Dict[str, Any]) -> Dict[str, Any]:
        relationship_type = person1.get("relationship_type", "")
        degree = self.calculate_degree(relationship_type)
        is_related = (degree <= CONFIG["MAX_FAMILY_DEGREE"]) if degree else False

        return {
            "isRelatedParty": is_related,
            "degree": degree,
            "relationshipType": relationship_type,
            "basis": f"Family relationship - {degree} degree" if is_related else "No family relation within 4th degree"
        }

    def calculate_degree(self, relationship_type: str) -> int:
        for deg, relations in self.family_degrees.items():
            if relationship_type in relations:
                return deg
        return 999  # Large number if not found


# -------------------------------------------------------------------------
# 4. Corporate Relationship Analyzer
# -------------------------------------------------------------------------
class CorporateRelationshipAnalyzer:
    """
    Analyzes ownership, control, and management to determine
    if two companies are Related Parties.
    """

    def assess_relationship(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> Dict[str, Any]:
        ownership_analysis = self.analyze_ownership(entity1, entity2)
        control_analysis = self.analyze_control(entity1, entity2)
        management_analysis = self.analyze_management(entity1, entity2)

        return {
            "isRelatedParty": any([
                ownership_analysis["isRelated"],
                control_analysis["isRelated"],
                management_analysis["isRelated"]
            ]),
            "relationships": {
                "ownership": ownership_analysis,
                "control": control_analysis,
                "management": management_analysis
            }
        }

    def analyze_ownership(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> Dict[str, Any]:
        direct_ownership_1 = entity1.get("ownership", {}).get("direct", 0)
        indirect_ownership_1 = entity1.get("ownership", {}).get("indirect", 0)
        total_ownership_1 = direct_ownership_1 + indirect_ownership_1

        direct_ownership_2 = entity2.get("ownership", {}).get("direct", 0)
        indirect_ownership_2 = entity2.get("ownership", {}).get("indirect", 0)
        total_ownership_2 = direct_ownership_2 + indirect_ownership_2

        is_related_1 = (total_ownership_1 >= CONFIG["OWNERSHIP_THRESHOLD"])
        is_related_2 = (total_ownership_2 >= CONFIG["OWNERSHIP_THRESHOLD"])

        return {
            "isRelated": is_related_1 or is_related_2,
            "entity1_total_ownership": total_ownership_1,
            "entity2_total_ownership": total_ownership_2,
            "basis": "Ownership >= 50%"
        }

    def analyze_control(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> Dict[str, Any]:
        entity1_voting = entity1.get("votingRights", 0)
        entity2_voting = entity2.get("votingRights", 0)
        entity1_profit = entity1.get("profitEntitlement", 0)
        entity2_profit = entity2.get("profitEntitlement", 0)

        is_related_1 = (entity1_voting >= CONFIG["CONTROL_THRESHOLD"]) or (entity1_profit >= CONFIG["CONTROL_THRESHOLD"])
        is_related_2 = (entity2_voting >= CONFIG["CONTROL_THRESHOLD"]) or (entity2_profit >= CONFIG["CONTROL_THRESHOLD"])

        return {
            "isRelated": is_related_1 or is_related_2,
            "entity1_voting": entity1_voting,
            "entity2_voting": entity2_voting,
            "entity1_profit": entity1_profit,
            "entity2_profit": entity2_profit,
            "basis": "Control through voting rights or profit entitlement >= 50%"
        }

    def analyze_management(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> Dict[str, Any]:
        entity1_control = entity1.get("managementControl", 0)
        entity2_control = entity2.get("managementControl", 0)

        is_related_1 = (entity1_control >= CONFIG["CONTROL_THRESHOLD"])
        is_related_2 = (entity2_control >= CONFIG["CONTROL_THRESHOLD"])

        return {
            "isRelated": is_related_1 or is_related_2,
            "entity1_management_control": entity1_control,
            "entity2_management_control": entity2_control,
            "basis": "Management control >= 50%"
        }


# -------------------------------------------------------------------------
# 5. Connected Person Analyzer (Individual -> Company)
# -------------------------------------------------------------------------
class ConnectedPersonAnalyzer:
    """
    Analyzes if an Individual is a Connected Person to a Company,
    based on director/officer roles, ownership, or significant influence.
    """

    def assess_connection(self, person: Dict[str, Any], entity: Dict[str, Any]) -> Dict[str, Any]:
        directorship_test = self.check_directorship(person, entity)
        officer_test = self.check_officer(person, entity)
        ownership_test = self.check_ownership(person, entity)

        is_connected = any([
            directorship_test["isConnected"],
            officer_test["isConnected"],
            ownership_test["isConnected"]
        ])

        basis = []
        if directorship_test["isConnected"]:
            basis.append(directorship_test["basis"])
        if officer_test["isConnected"]:
            basis.append(officer_test["basis"])
        if ownership_test["isConnected"]:
            basis.append(ownership_test["basis"])

        return {
            "isConnectedPerson": is_connected,
            "basis": basis
        }

    def check_directorship(self, person: Dict[str, Any], entity: Dict[str, Any]) -> Dict[str, Any]:
        is_director = person.get("isDirector", False)
        is_connected = bool(is_director)
        return {
            "isConnected": is_connected,
            "basis": "Director relationship" if is_connected else ""
        }

    def check_officer(self, person: Dict[str, Any], entity: Dict[str, Any]) -> Dict[str, Any]:
        is_officer = person.get("isOfficer", False)
        is_connected = bool(is_officer)
        return {
            "isConnected": is_connected,
            "basis": "Officer relationship" if is_connected else ""
        }

    def check_ownership(self, person: Dict[str, Any], entity: Dict[str, Any]) -> Dict[str, Any]:
        personal_ownership = person.get("personalOwnership", 0)
        family_ownership = person.get("familyOwnership", 0)
        combined_ownership = personal_ownership + family_ownership

        is_connected = (combined_ownership >= CONFIG["OWNERSHIP_THRESHOLD"])
        return {
            "isConnected": is_connected,
            "basis": "Ownership >= 50%" if is_connected else ""
        }


# -------------------------------------------------------------------------
# 6. Special Entity Analyzer (Permanent Establishment, etc.)
# -------------------------------------------------------------------------
class SpecialEntityAnalyzer:
    """
    Analyzes special cases like Permanent Establishments, Partnerships,
    Trusts/Foundations, etc., to see if they are automatically related.
    """

    def assess_permanent_establishment(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
        if entity1.get("type") == "CORPORATE" and entity2.get("type") == "PERMANENT_ESTABLISHMENT":
            return True
        return False

    def assess_partnership(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
        if entity1.get("companyType") == "Partnership" and entity2.get("companyType") == "Partnership":
            return True
        return False

    def assess_trust_foundation(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
        if entity1.get("companyType") in ["Trust", "Foundation"] or entity2.get("companyType") in ["Trust", "Foundation"]:
            return True
        return False


# -------------------------------------------------------------------------
# 7. Risk Analyzer
# -------------------------------------------------------------------------
class RiskAnalyzer:
    """
    Evaluates the risk level of the relationship based on complexity, 
    cross-border transactions, etc.
    """

    def analyze_risk(self, relationship_data: Dict[str, Any]) -> Dict[str, Any]:
        complexity_score = self.assess_complexity(relationship_data)
        cross_border = relationship_data.get("crossBorder", False)
        high_value = relationship_data.get("highValueTransactions", False)

        risk_level = "LOW"
        if complexity_score >= 3 or cross_border or high_value:
            risk_level = "HIGH"
        elif complexity_score == 2:
            risk_level = "MEDIUM"

        return {
            "riskLevel": risk_level,
            "factors": {
                "complexityScore": complexity_score,
                "crossBorder": cross_border,
                "highValueTransactions": high_value
            }
        }

    def assess_complexity(self, relationship_data: Dict[str, Any]) -> int:
        layers = relationship_data.get("layersOfOwnership", 1)
        if layers > 3:
            return 3
        elif layers == 3:
            return 2
        else:
            return 1


# -------------------------------------------------------------------------
# 8. Documentation Analyzer
# -------------------------------------------------------------------------
class DocumentationAnalyzer:
    """
    Determines the required documentation (Master File, Local File, 
    Disclosure Form) based on revenue thresholds and transaction materiality.
    """

    def determine_requirements(self, entity: Dict[str, Any], relationship: Dict[str, Any], risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        master_file_req = self.is_master_file_required(entity)
        local_file_req = self.is_local_file_required(entity)
        disclosure_form_req = self.is_disclosure_required(relationship)

        additional_docs = []
        if risk_assessment["riskLevel"] == "HIGH":
            additional_docs.append("Detailed Intercompany Agreement Documentation")

        return {
            "masterFile": master_file_req,
            "localFile": local_file_req,
            "disclosureForm": disclosure_form_req,
            "additionalDocs": additional_docs
        }

    def is_master_file_required(self, entity: Dict[str, Any]) -> bool:
        group_revenue = entity.get("groupConsolidatedRevenue", 0)
        return group_revenue >= CONFIG["DOCUMENTATION_THRESHOLDS"]["MNE_GROUP_CONSOLIDATED"]

    def is_local_file_required(self, entity: Dict[str, Any]) -> bool:
        revenue = entity.get("revenue", 0)
        return revenue >= CONFIG["DOCUMENTATION_THRESHOLDS"]["REVENUE"]

    def is_disclosure_required(self, relationship: Dict[str, Any]) -> bool:
        return bool(relationship.get("isRelatedParty", False))


# -------------------------------------------------------------------------
# 9. Main Assessment Controller
# -------------------------------------------------------------------------
class TPRelationshipAssessor:
    """
    Orchestrates all analyzers and returns a final result dict including:
    - Whether parties are Related Parties
    - Whether they are Connected Persons
    - Basis for determination
    - Special Cases
    - Documentation requirements
    - Risk assessment
    """

    def __init__(self):
        self.family_analyzer = FamilyRelationshipAnalyzer()
        self.corporate_analyzer = CorporateRelationshipAnalyzer()
        self.connected_analyzer = ConnectedPersonAnalyzer()
        self.special_entity_analyzer = SpecialEntityAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.doc_analyzer = DocumentationAnalyzer()

    def assess_relationship(self, party1: Dict[str, Any], party2: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            "isRelatedParty": False,
            "isConnectedPerson": False,
            "basis": [],
            "specialCases": [],
            "riskAssessment": {},
            "documentationRequired": {}
        }

        party1_type = party1.get("type", "").upper()
        party2_type = party2.get("type", "").upper()

        # INDIVIDUAL to INDIVIDUAL
        if party1_type == "INDIVIDUAL" and party2_type == "INDIVIDUAL":
            fam_rel = self.family_analyzer.assess_relationship(party1, party2)
            if fam_rel["isRelatedParty"]:
                result["isRelatedParty"] = True
                result["basis"].append(fam_rel["basis"])

        # CORPORATE to CORPORATE
        elif party1_type == "CORPORATE" and party2_type == "CORPORATE":
            corp_rel = self.corporate_analyzer.assess_relationship(party1, party2)
            if corp_rel["isRelatedParty"]:
                result["isRelatedParty"] = True
                for k, v in corp_rel["relationships"].items():
                    if v["isRelated"]:
                        result["basis"].append(v["basis"])

        # INDIVIDUAL to CORPORATE => Check connected person + ownership
        elif (party1_type == "INDIVIDUAL" and party2_type == "CORPORATE") or \
             (party1_type == "CORPORATE" and party2_type == "INDIVIDUAL"):
            if party1_type == "INDIVIDUAL":
                connected_rel = self.connected_analyzer.assess_connection(party1, party2)
            else:
                connected_rel = self.connected_analyzer.assess_connection(party2, party1)

            corp_analyzer_result = self.corporate_analyzer.assess_relationship(party1, party2)

            if corp_analyzer_result["isRelatedParty"]:
                result["isRelatedParty"] = True
                for k, v in corp_analyzer_result["relationships"].items():
                    if v["isRelated"]:
                        result["basis"].append(v["basis"])

            if connected_rel["isConnectedPerson"]:
                result["isConnectedPerson"] = True
                for b in connected_rel["basis"]:
                    result["basis"].append(b)

        # Special entity checks
        if self.special_entity_analyzer.assess_permanent_establishment(party1, party2):
            result["isRelatedParty"] = True
            result["specialCases"].append("Permanent Establishment relationship")

        if self.special_entity_analyzer.assess_partnership(party1, party2):
            result["isRelatedParty"] = True
            result["specialCases"].append("Partnership relationship")

        if self.special_entity_analyzer.assess_trust_foundation(party1, party2):
            result["isRelatedParty"] = True
            result["specialCases"].append("Trust/Foundation relationship")

        # Risk analysis
        risk_data = {
            "crossBorder": party1.get("crossBorder", False) or party2.get("crossBorder", False),
            "highValueTransactions": party1.get("highValueTransactions", False) or party2.get("highValueTransactions", False),
            "layersOfOwnership": max(party1.get("layersOfOwnership", 1), party2.get("layersOfOwnership", 1))
        }
        risk_assessment = self.risk_analyzer.analyze_risk(risk_data)
        result["riskAssessment"] = risk_assessment

        # Documentation
        doc_req = self.doc_analyzer.determine_requirements(party1, result, risk_assessment)
        result["documentationRequired"] = doc_req

        return result


# -------------------------------------------------------------------------
# 10. Streamlit Front-End
# -------------------------------------------------------------------------
def main():
    st.title("UAE Related Party & Connected Person AI Assessment Tool")

    st.markdown(
        """
        This tool determines if two parties are Related Parties or Connected Persons
        under UAE Transfer Pricing regulations. 
        Provide information for each party below:
        """
    )

    # Party 1 Inputs
    st.subheader("Party 1 Information")
    party1_type = st.selectbox("Type of Party 1", ["INDIVIDUAL", "CORPORATE", "PERMANENT_ESTABLISHMENT"])
    cross_border_1 = st.checkbox("Cross-border transactions for Party 1?", False)
    high_value_1 = st.checkbox("High-value transactions for Party 1?", False)
    layers_1 = st.number_input("Layers of Ownership (Party 1)", min_value=1, max_value=10, value=1)

    party1 = {
        "type": party1_type,
        "crossBorder": cross_border_1,
        "highValueTransactions": high_value_1,
        "layersOfOwnership": layers_1
    }

    # If Party 1 is INDIVIDUAL
    if party1_type == "INDIVIDUAL":
        relationship_type = st.selectbox("Family Relationship Type (if relevant)", [
            "", "parent", "child", "spouse_parent", "spouse_child",
            "grandparent", "grandchild", "sibling", "spouse_sibling",
            "uncle", "aunt", "niece", "nephew", "first_cousin"
        ])
        is_director = st.checkbox("Is Director? (Party 1)", False)
        is_officer = st.checkbox("Is Officer? (Party 1)", False)
        personal_ownership = st.number_input("Personal Ownership % (0-100, Party 1)", 0, 100, 0)
        family_ownership = st.number_input("Family Ownership % (0-100, Party 1)", 0, 100, 0)

        party1["relationship_type"] = relationship_type
        party1["isDirector"] = is_director
        party1["isOfficer"] = is_officer
        party1["personalOwnership"] = personal_ownership
        party1["familyOwnership"] = family_ownership

    # If Party 1 is CORPORATE
    if party1_type == "CORPORATE":
        revenue_1 = st.number_input("Annual Revenue (Party 1)", min_value=0, value=0)
        group_rev_1 = st.number_input("Group Consolidated Revenue (Party 1)", min_value=0, value=0)
        direct_own_1 = st.number_input("Direct Ownership % (Party 1)", 0, 100, 0)
        indirect_own_1 = st.number_input("Indirect Ownership % (Party 1)", 0, 100, 0)
        voting_rights_1 = st.number_input("Voting Rights % (Party 1)", 0, 100, 0)
        profit_entitlement_1 = st.number_input("Profit Entitlement % (Party 1)", 0, 100, 0)
        management_control_1 = st.number_input("Management Control % (Party 1)", 0, 100, 0)

        party1["revenue"] = revenue_1
        party1["groupConsolidatedRevenue"] = group_rev_1
        party1["ownership"] = {"direct": direct_own_1, "indirect": indirect_own_1}
        party1["votingRights"] = voting_rights_1
        party1["profitEntitlement"] = profit_entitlement_1
        party1["managementControl"] = management_control_1

    # Party 2 Inputs
    st.subheader("Party 2 Information")
    party2_type = st.selectbox("Type of Party 2", ["INDIVIDUAL", "CORPORATE", "PERMANENT_ESTABLISHMENT"])
    cross_border_2 = st.checkbox("Cross-border transactions for Party 2?", False)
    high_value_2 = st.checkbox("High-value transactions for Party 2?", False)
    layers_2 = st.number_input("Layers of Ownership (Party 2)", min_value=1, max_value=10, value=1)

    party2 = {
        "type": party2_type,
        "crossBorder": cross_border_2,
        "highValueTransactions": high_value_2,
        "layersOfOwnership": layers_2
    }

    # If Party 2 is INDIVIDUAL
    if party2_type == "INDIVIDUAL":
        relationship_type_2 = st.selectbox("Family Relationship Type (if relevant for Party 2)", [
            "", "parent", "child", "spouse_parent", "spouse_child",
            "grandparent", "grandchild", "sibling", "spouse_sibling",
            "uncle", "aunt", "niece", "nephew", "first_cousin"
        ])
        is_director_2 = st.checkbox("Is Director? (Party 2)", False)
        is_officer_2 = st.checkbox("Is Officer? (Party 2)", False)
        personal_ownership_2 = st.number_input("Personal Ownership % (Party 2)", 0, 100, 0)
        family_ownership_2 = st.number_input("Family Ownership % (Party 2)", 0, 100, 0)

        party2["relationship_type"] = relationship_type_2
        party2["isDirector"] = is_director_2
        party2["isOfficer"] = is_officer_2
        party2["personalOwnership"] = personal_ownership_2
        party2["familyOwnership"] = family_ownership_2

    # If Party 2 is CORPORATE
    if party2_type == "CORPORATE":
        revenue_2 = st.number_input("Annual Revenue (Party 2)", min_value=0, value=0)
        group_rev_2 = st.number_input("Group Consolidated Revenue (Party 2)", min_value=0, value=0)
        direct_own_2 = st.number_input("Direct Ownership % (Party 2)", 0, 100, 0)
        indirect_own_2 = st.number_input("Indirect Ownership % (Party 2)", 0, 100, 0)
        voting_rights_2 = st.number_input("Voting Rights % (Party 2)", 0, 100, 0)
        profit_entitlement_2 = st.number_input("Profit Entitlement % (Party 2)", 0, 100, 0)
        management_control_2 = st.number_input("Management Control % (Party 2)", 0, 100, 0)

        party2["revenue"] = revenue_2
        party2["groupConsolidatedRevenue"] = group_rev_2
        party2["ownership"] = {"direct": direct_own_2, "indirect": indirect_own_2}
        party2["votingRights"] = voting_rights_2
        party2["profitEntitlement"] = profit_entitlement_2
        party2["managementControl"] = management_control_2

    # Action: Assess Relationship
    if st.button("Assess Relationship"):
        assessor = TPRelationshipAssessor()
        assessment_result = assessor.assess_relationship(party1, party2)

        st.subheader("Assessment Result")
        st.write("Are they Related Parties?", assessment_result["isRelatedParty"])
        st.write("Are they Connected Persons?", assessment_result["isConnectedPerson"])
        st.write("Basis for Determination:", assessment_result["basis"])
        
        if assessment_result["specialCases"]:
            st.write("Special Cases Identified:", assessment_result["specialCases"])

        st.subheader("Risk Assessment")
        st.write("Risk Level:", assessment_result["riskAssessment"].get("riskLevel"))
        st.write("Risk Factors:", assessment_result["riskAssessment"].get("factors"))

        st.subheader("Documentation Requirements")
        doc_req = assessment_result["documentationRequired"]
        st.write("Master File Required:", doc_req.get("masterFile", False))
        st.write("Local File Required:", doc_req.get("localFile", False))
        st.write("Disclosure Form Required:", doc_req.get("disclosureForm", False))
        if doc_req.get("additionalDocs"):
            st.write("Additional Documentation Needed:", doc_req["additionalDocs"])

if __name__ == "__main__":
    main()
