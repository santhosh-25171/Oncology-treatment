"""
Oncology Terminology Dictionary and Normalization Engine for Stage 6 Agentic AI.

Provides comprehensive mapping, canonical identifiers, synonym expansion, and entity
detection across cancer types, genes, mutations, biomarkers, drugs, adverse events,
treatment-response metrics, progression patterns, and resistance mechanisms.

Preserves the original clinical narrative while resolving clinical synonyms into standard
ontology identifiers.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple
from personalized_precision_oncology.stage6_agentic.agentic.knowledge.schemas import TermMatch


class OncologyDictionary:
    """
    Deterministic dictionary and normalizer for precision oncology terminology.
    Maintains zero mutable state during lookups to guarantee thread safety.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, TermMatch] = {}
        self._alias_map: Dict[str, str] = {}
        self._build_dictionary()

    def _register(
        self,
        canonical_id: str,
        category: str,
        canonical_name: str,
        synonyms: List[str],
        description: Optional[str] = None
    ) -> None:
        """Register a canonical term with its aliases in the lookup tables."""
        match_obj = TermMatch(
            canonical_id=canonical_id,
            category=category,
            canonical_name=canonical_name,
            matched_text=canonical_name,
            synonyms=synonyms,
            description=description
        )
        self._entries[canonical_id] = match_obj

        # Register canonical lower
        self._alias_map[canonical_name.lower()] = canonical_id
        # Register synonyms
        for syn in synonyms:
            self._alias_map[syn.lower()] = canonical_id

    def _build_dictionary(self) -> None:
        """Build the master vocabulary."""
        # 1. CANCER TYPES
        self._register(
            "CANCER:NSCLC", "cancer_type", "Non-Small Cell Lung Cancer",
            ["nsclc", "non-small cell lung carcinoma", "lung carcinoma", "lung adenocarcinoma", "luad", "lusc", "squamous cell lung cancer", "squamous cell carcinoma of the lung"],
            "Malignant epithelial tumor of the lung excluding small cell carcinoma."
        )
        self._register(
            "CANCER:BREAST", "cancer_type", "Breast Cancer",
            ["breast carcinoma", "invasive ductal carcinoma", "idc", "tnbc", "triple negative breast cancer", "her2+ breast cancer", "hr+ breast cancer"],
            "Malignancy originating in breast tissue."
        )
        self._register(
            "CANCER:CRC", "cancer_type", "Colorectal Cancer",
            ["crc", "colorectal carcinoma", "colon cancer", "rectal cancer", "colon adenocarcinoma"],
            "Adenocarcinoma of the colon or rectum."
        )
        self._register(
            "CANCER:SCLC", "cancer_type", "Small Cell Lung Cancer",
            ["sclc", "small cell neuroendocrine carcinoma of the lung", "oat cell carcinoma"],
            "Aggressive high-grade neuroendocrine lung carcinoma."
        )

        # 2. GENES
        self._register("GENE:EGFR", "gene", "EGFR", ["epidermal growth factor receptor", "erbb1", "her1"], "Receptor tyrosine kinase frequently altered in NSCLC.")
        self._register("GENE:KRAS", "gene", "KRAS", ["k-ras", "kras proto-oncogene"], "GTPase signal transducer regulating MAPK signaling.")
        self._register("GENE:ALK", "gene", "ALK", ["anaplastic lymphoma kinase", "eml4-alk"], "Receptor tyrosine kinase rearranged in 3-5% of NSCLC.")
        self._register("GENE:BRAF", "gene", "BRAF", ["b-raf", "braf proto-oncogene"], "Serine/threonine kinase downstream of KRAS in MAPK pathway.")
        self._register("GENE:MET", "gene", "MET", ["c-met", "hepatocyte growth factor receptor", "hgfr"], "Receptor tyrosine kinase associated with exon 14 skipping or amplification.")
        self._register("GENE:ERBB2", "gene", "ERBB2", ["her2", "neu", "her2/neu"], "Receptor tyrosine kinase amplified in breast/gastric or mutated in NSCLC.")
        self._register("GENE:RET", "gene", "RET", ["ret proto-oncogene"], "Receptor tyrosine kinase rearranged in NSCLC and thyroid cancer.")
        self._register("GENE:ROS1", "gene", "ROS1", ["ros1 proto-oncogene"], "Receptor tyrosine kinase rearranged in NSCLC.")
        self._register("GENE:NTRK1", "gene", "NTRK1", ["trka", "neurotrophic receptor tyrosine kinase 1"], "Neurotrophin receptor kinase subject to oncogenic fusions.")
        self._register("GENE:STK11", "gene", "STK11", ["lkb1", "serine/threonine kinase 11"], "Tumor suppressor loss linked to primary immunotherapy resistance.")
        self._register("GENE:KEAP1", "gene", "KEAP1", ["kelch-like ech-associated protein 1"], "Regulator of NRF2 oxidative stress pathway; confers cold tumor microenvironment.")
        self._register("GENE:TP53", "gene", "TP53", ["p53", "tumor protein p53"], "Master tumor suppressor genome guardian.")
        self._register("GENE:PIK3CA", "gene", "PIK3CA", ["pik3ca", "phosphatidylinositol-4,5-bisphosphate 3-kinase catalytic subunit alpha"], "PI3K/AKT/mTOR pathway oncogene.")
        self._register("GENE:RB1", "gene", "RB1", ["retinoblastoma 1", "prb"], "Cell cycle tumor suppressor; loss associated with neuroendocrine transformation.")
        self._register("GENE:BRCA1", "gene", "BRCA1", ["breast cancer 1"], "Homologous recombination repair gene.")
        self._register("GENE:BRCA2", "gene", "BRCA2", ["breast cancer 2"], "Homologous recombination repair gene.")

        # 3. MUTATIONS & VARIANTS
        self._register("MUT:EGFR_L858R", "mutation", "EGFR L858R", ["l858r", "p.l858r", "exon 21 l858r", "egfr l858r missense"], "Canonical sensitizing exon 21 missense alteration.")
        self._register("MUT:EGFR_EX19DEL", "mutation", "EGFR Exon 19 del", ["exon 19 deletion", "e746_a750del", "p.e746_a750del", "ex19del", "egfr ex19del"], "Canonical sensitizing in-frame deletion.")
        self._register("MUT:EGFR_T790M", "mutation", "EGFR T790M", ["t790m", "p.t790m", "exon 20 t790m", "egfr t790m"], "Exon 20 gatekeeper resistance alteration conferring steric hindrance to 1st/2nd gen TKIs.")
        self._register("MUT:EGFR_C797S", "mutation", "EGFR C797S", ["c797s", "p.c797s", "c797s in cis", "c797s in trans", "egfr c797s"], "Exon 20 tertiary mutation disrupting covalent binding of 3rd gen TKIs (osimertinib).")
        self._register("MUT:EGFR_L718Q", "mutation", "EGFR L718Q", ["l718q", "p.l718q"], "ATP binding pocket mutation conferring resistance to 3rd/4th gen TKIs.")
        self._register("MUT:KRAS_G12C", "mutation", "KRAS G12C", ["g12c", "p.g12c", "kras g12c missense"], "Activating GTPase codon 12 mutation targetable by sotorasib/adagrasib.")
        self._register("MUT:KRAS_G12D", "mutation", "KRAS G12D", ["g12d", "p.g12d"], "Activating GTPase codon 12 aspartate substitution.")
        self._register("MUT:KRAS_G12V", "mutation", "KRAS G12V", ["g12v", "p.g12v"], "Activating GTPase codon 12 valine substitution.")
        self._register("MUT:KRAS_G13D", "mutation", "KRAS G13D", ["g13d", "p.g13d"], "Activating GTPase codon 13 substitution.")
        self._register("MUT:KRAS_Y99C", "mutation", "KRAS Y99C", ["y99c", "p.y99c"], "Switch II pocket secondary resistance mutation to covalent G12C inhibitors.")
        self._register("MUT:ALK_G1202R", "mutation", "ALK G1202R", ["g1202r", "p.g1202r"], "Solvent front bulky kinase domain resistance mutation sensitive to lorlatinib.")
        self._register("MUT:ALK_L1196M", "mutation", "ALK L1196M", ["l1196m", "p.l1196m"], "ALK gatekeeper resistance mutation.")
        self._register("MUT:ALK_FUSION", "mutation", "EML4-ALK", ["eml4-alk fusion", "alk rearrangement", "alk fusion"], "Canonical oncogenic inversion rearrangement.")
        self._register("MUT:BRAF_V600E", "mutation", "BRAF V600E", ["v600e", "p.v600e"], "Activating kinase domain missense mutation sensitive to BRAF/MEK inhibitors.")
        self._register("MUT:MET_EX14", "mutation", "MET Exon 14 Skipping", ["met exon 14", "exon 14 splice site", "d1010h", "p.d1010h"], "Juxtamembrane domain deletion impairing CBL ubiquitin ligase degradation.")
        self._register("MUT:ERBB2_EX20INS", "mutation", "ERBB2 Exon 20 In-Frame Insertion", ["erbb2 exon 20", "her2 exon 20 insertion", "y772_a775dup", "p.y772_a775dup"], "Kinase domain insertion sensitive to trastuzumab deruxtecan.")

        # 4. BIOMARKERS
        self._register("BIO:TMB", "biomarker", "TMB", ["tumor mutational burden", "mutational burden", "tmb-high", "tmb-low", "tmb_status"], "Total non-synonymous somatic mutations per coding megabase.")
        self._register("BIO:PDL1", "biomarker", "PD-L1", ["pdl1", "pd-l1 tps", "tumor proportion score", "tps", "cps", "combined positive score"], "Programmed death-ligand 1 cell-surface expression.")
        self._register("BIO:CTDNA", "biomarker", "ctDNA", ["circulating tumor dna", "ctdna maf", "mutant allele fraction", "maf", "liquid biopsy"], "Cell-free tumor-derived DNA in peripheral blood.")
        self._register("BIO:MSI", "biomarker", "MSI Status", ["msi", "microsatellite instability", "msi-h", "mss", "dmr", "pmmr"], "Genomic instability from defective DNA mismatch repair.")
        self._register("BIO:NLR", "biomarker", "NLR", ["neutrophil-to-lymphocyte ratio", "neutrophil lymphocyte ratio"], "Systemic inflammation marker reflecting immune balance.")
        self._register("BIO:CRP", "biomarker", "CRP", ["c-reactive protein"], "Acute phase reactant elevated in tumor-driven systemic inflammation.")
        self._register("BIO:LDH", "biomarker", "LDH", ["lactate dehydrogenase"], "Enzyme elevated in high metabolic tumor turnover and necrosis.")

        # 5. DRUGS
        self._register("DRUG:OSIMERTINIB", "drug", "Osimertinib", ["tagrisso", "azd9291"], "Third-generation CNS-active irreversible EGFR TKI.")
        self._register("DRUG:ERLOTINIB", "drug", "Erlotinib", ["tarceva"], "First-generation reversible EGFR TKI.")
        self._register("DRUG:GEFITINIB", "drug", "Gefitinib", ["iressa"], "First-generation reversible EGFR TKI.")
        self._register("DRUG:AFATINIB", "drug", "Afatinib", ["gilotrif"], "Second-generation irreversible ErbB family TKI.")
        self._register("DRUG:SOTORASIB", "drug", "Sotorasib", ["lumakras", "amg 510"], "First-in-class small-molecule covalent KRAS G12C inhibitor.")
        self._register("DRUG:ADAGRASIB", "drug", "Adagrasib", ["krazati", "mrtx849"], "Potent covalent KRAS G12C inhibitor.")
        self._register("DRUG:ALECTINIB", "drug", "Alectinib", ["alecensa"], "Second-generation CNS-penetrant ALK TKI.")
        self._register("DRUG:CRIZOTINIB", "drug", "Crizotinib", ["xalkori"], "First-generation multi-targeted ALK/ROS1/MET inhibitor.")
        self._register("DRUG:LORLATINIB", "drug", "Lorlatinib", ["lorbrena"], "Third-generation brain-penetrant ALK/ROS1 inhibitor active against G1202R.")
        self._register("DRUG:BRIGATINIB", "drug", "Brigatinib", ["alunbrig"], "Next-generation ALK and EGFR TKI.")
        self._register("DRUG:PEMBROLIZUMAB", "drug", "Pembrolizumab", ["keytruda"], "Humanized IgG4 anti-PD-1 monoclonal checkpoint inhibitor.")
        self._register("DRUG:NIVOLUMAB", "drug", "Nivolumab", ["opdivo"], "Fully human IgG4 anti-PD-1 monoclonal antibody.")
        self._register("DRUG:IPILIMUMAB", "drug", "Ipilimumab", ["yervoy"], "Human IgG1 anti-CTLA-4 monoclonal checkpoint inhibitor.")
        self._register("DRUG:CISPLATIN", "drug", "Cisplatin", ["platinol", "cddp"], "DNA crosslinking platinum chemotherapeutic agent.")
        self._register("DRUG:CARBOPLATIN", "drug", "Carboplatin", ["paraplatin"], "Second-generation platinum antineoplastic agent.")
        self._register("DRUG:PACLITAXEL", "drug", "Paclitaxel", ["taxol"], "Microtubule stabilizing taxane chemotherapeutic.")
        self._register("DRUG:PEMETREXED", "drug", "Pemetrexed", ["alimta"], "Multitargeted antifolate antimetabolite.")
        self._register("DRUG:TRASTUZUMAB", "drug", "Trastuzumab", ["herceptin"], "Humanized anti-HER2 monoclonal antibody.")
        self._register("DRUG:TDXD", "drug", "Trastuzumab Deruxtecan", ["enhertu", "t-dxd", "ds-8201"], "HER2-directed antibody-drug conjugate with topoisomerase I inhibitor.")
        self._register("DRUG:DABRAFENIB", "drug", "Dabrafenib", ["tafinlar"], "Potent selective BRAF kinase inhibitor.")
        self._register("DRUG:TRAMETINIB", "drug", "Trametinib", ["mekinist"], "Reversible allosteric MEK1/MEK2 kinase inhibitor.")
        self._register("DRUG:CAPMATINIB", "drug", "Capmatinib", ["tabrecta"], "Potent selective MET kinase inhibitor.")
        self._register("DRUG:TEPOTINIB", "drug", "Tepotinib", ["tepmetko"], "Oral selective MET kinase inhibitor.")
        self._register("DRUG:SELPERCATINIB", "drug", "Selpercatinib", ["retevmo"], "Selective CNS-active RET kinase inhibitor.")
        self._register("DRUG:5FU", "drug", "Fluorouracil", ["5-fu", "adrucil"], "Thymidylate synthase pyrimidine analog antimetabolite.")
        self._register("DRUG:CAPECITABINE", "drug", "Capecitabine", ["xeloda"], "Oral prodrug of fluorouracil.")
        self._register("DRUG:IRINOTECAN", "drug", "Irinotecan", ["camptosar", "cpt-11"], "Topoisomerase I inhibitor.")
        self._register("DRUG:OXALIPLATIN", "drug", "Oxaliplatin", ["eloxatin"], "Third-generation diaminocyclohexane platinum agent.")

        # 6. ADVERSE EVENTS & TOXICITY
        self._register("AE:GENERAL_TOXICITY", "adverse_event", "Toxicity", ["adverse event", "toxicity", "toxicities", "high toxicity", "severe toxicity"], "Undesirable secondary physiological effect of therapy.")
        self._register("AE:NAUSEA", "adverse_event", "Nausea", ["nausea", "severe nausea", "queasiness"], "Feeling of sickness with an inclination to vomit.")
        self._register("AE:VOMITING", "adverse_event", "Vomiting", ["vomiting", "emesis"], "Oral ejection of gastrointestinal contents.")
        self._register("AE:NEUTROPENIA", "adverse_event", "Neutropenia", ["neutropenia", "febrile neutropenia", "low absolute neutrophil count", "anc depression"], "Abnormally low concentration of neutrophils.")
        self._register("AE:NEPHROTOXICITY", "adverse_event", "Nephrotoxicity", ["nephrotoxicity", "renal toxicity", "acute kidney injury", "aki", "creatinine elevation", "renal impairment"], "Toxicity damaging kidneys.")
        self._register("AE:HEPATOTOXICITY", "adverse_event", "Hepatotoxicity", ["hepatotoxicity", "liver toxicity", "ast elevation", "alt elevation", "transaminitis"], "Chemical-driven liver injury.")
        self._register("AE:DIARRHEA", "adverse_event", "Diarrhea", ["diarrhea", "colitis", "loose stools"], "Frequent watery bowel evacuations.")
        self._register("AE:NEUROPATHY", "adverse_event", "Peripheral Neuropathy", ["neuropathy", "peripheral neuropathy", "neurotoxicity", "paresthesia"], "Damage to peripheral sensory nerves.")
        self._register("AE:RASH", "adverse_event", "Rash", ["rash", "acneiform rash", "maculopapular rash", "dermatitis"], "Eruption of skin lesions.")
        self._register("AE:PNEUMONITIS", "adverse_event", "Pneumonitis", ["pneumonitis", "interstitial lung disease", "ild", "pulmonary toxicity"], "Inflammation of lung alveolar walls.")
        self._register("AE:CARDIOTOXICITY", "adverse_event", "Cardiotoxicity", ["cardiotoxicity", "qt prolongation", "qtc prolongation", "lvef decline", "heart failure"], "Damage to heart muscle or electrical conduction.")
        self._register("AE:IRAE", "adverse_event", "Immune-Related Adverse Event", ["irae", "immune-mediated toxicity", "immune toxicity"], "Autoimmune inflammatory toxicity from checkpoint inhibitors.")

        # 7. TREATMENT RESPONSE & PROGRESSION
        self._register("RESP:RESPONDER", "response", "Responder", ["responder", "favorable response", "treatment responder"], "Patient demonstrating objective tumor regression.")
        self._register("RESP:NON_RESPONDER", "response", "Non-Responder", ["non-responder", "treatment non-responder", "refractory", "non-responsive"], "Patient failing to exhibit objective clinical benefit.")
        self._register("RESP:CR", "response", "Complete Response", ["complete response", "cr", "complete remission"], "Disappearance of all target and non-target lesions.")
        self._register("RESP:PR", "response", "Partial Response", ["partial response", "pr", "partial remission"], ">=30% decrease in sum of longest target lesion diameters.")
        self._register("RESP:SD", "response", "Stable Disease", ["stable disease", "sd"], "Neither sufficient shrinkage for PR nor sufficient increase for PD.")
        self._register("RESP:PD", "progression", "Progressive Disease", ["progressive disease", "progression", "disease progression", "recist progression", "tumor progression"], ">=20% increase in sum of target diameters or appearance of new lesions.")
        self._register("RESP:METASTASIS", "progression", "Metastasis", ["metastatic progression", "distant metastasis", "metastases", "organ spread"], "Spread of cancer cells from primary site to distant organs.")
        self._register("RESP:CNS_PROGRESSION", "progression", "CNS Progression", ["cns progression", "brain metastases", "cerebellar metastasis", "leptomeningeal disease"], "Progression of disease in the central nervous system.")

        # 8. RESISTANCE CONCEPTS
        self._register("RES:ACQUIRED", "resistance", "Acquired Resistance", ["acquired resistance", "secondary resistance"], "Tumor growth after initial objective response to targeted or systemic therapy.")
        self._register("RES:INTRINSIC", "resistance", "Intrinsic Resistance", ["intrinsic resistance", "primary resistance", "de novo resistance"], "Lack of initial response despite presence of target alteration.")
        self._register("RES:GATEKEEPER", "resistance", "Gatekeeper Mutation", ["gatekeeper", "gatekeeper mutation"], "Kinase domain alteration sterically hindering drug access to ATP binding pocket.")
        self._register("RES:SOLVENT_FRONT", "resistance", "Solvent-Front Mutation", ["solvent-front", "solvent front mutation", "solvent-front alteration"], "Mutation at solvent-exposed border sterically impeding bulky inhibitors.")
        self._register("RES:BYPASS", "resistance", "Bypass Pathway Activation", ["bypass resistance", "bypass pathway", "bypass rtk", "met amplification bypass"], "Activation of alternative receptor or parallel downstream pathway circumventing blocked target.")
        self._register("RES:SCLC_SWITCH", "resistance", "Histological SCLC Transformation", ["histological transformation", "sclc transformation", "small cell transformation", "lineage plasticity"], "Phenotypic shift from adenocarcinoma to small cell neuroendocrine carcinoma.")

    def normalize_term(self, term: str) -> Optional[TermMatch]:
        """
        Normalize an input term or phrase into its canonical TermMatch.
        Returns None if term is unknown.
        """
        cleaned = term.strip().lower()
        # Direct lookup in alias map
        if cleaned in self._alias_map:
            canonical_id = self._alias_map[cleaned]
            entry = self._entries[canonical_id]
            return TermMatch(
                canonical_id=entry.canonical_id,
                category=entry.category,
                canonical_name=entry.canonical_name,
                matched_text=term,
                synonyms=entry.synonyms,
                description=entry.description
            )
        # Regex cleanup for common prefixes/punctuation (e.g. "p.l858r" vs "l858r")
        stripped = re.sub(r"^[pP]\.", "", cleaned)
        if stripped in self._alias_map:
            canonical_id = self._alias_map[stripped]
            entry = self._entries[canonical_id]
            return TermMatch(
                canonical_id=entry.canonical_id,
                category=entry.category,
                canonical_name=entry.canonical_name,
                matched_text=term,
                synonyms=entry.synonyms,
                description=entry.description
            )
        return None

    def extract_normalized_entities(self, text: str) -> List[TermMatch]:
        """
        Scan text for all known oncology terms and return normalized entities.
        Does NOT alter the original text string.
        """
        if not text:
            return []

        lower_text = text.lower()
        matches: List[TermMatch] = []
        seen_canonical: Set[str] = set()

        # Sort alias keys by length descending to match longest phrases first (e.g. 'egfr l858r' before 'egfr')
        sorted_aliases = sorted(self._alias_map.keys(), key=len, reverse=True)

        for alias in sorted_aliases:
            if len(alias) < 2:
                continue
            # Word boundary search
            pattern = r"\b" + re.escape(alias) + r"\b"
            for m in re.finditer(pattern, lower_text):
                canonical_id = self._alias_map[alias]
                if canonical_id not in seen_canonical:
                    seen_canonical.add(canonical_id)
                    entry = self._entries[canonical_id]
                    matched_slice = text[m.start():m.end()]
                    matches.append(
                        TermMatch(
                            canonical_id=entry.canonical_id,
                            category=entry.category,
                            canonical_name=entry.canonical_name,
                            matched_text=matched_slice,
                            synonyms=entry.synonyms,
                            description=entry.description
                        )
                    )

        return matches

    def expand_query_terms(self, query: str) -> Set[str]:
        """
        Expand a search query into its normalized canonical names and key synonyms.
        Used by the deterministic lexical retriever.
        """
        expanded: Set[str] = set()
        # Add basic tokens
        tokens = [t.strip().lower() for t in re.split(r"[\s,;+\-/]+", query) if len(t.strip()) > 1]
        for t in tokens:
            expanded.add(t)

        # Detect recognized entities
        detected = self.extract_normalized_entities(query)
        for d in detected:
            expanded.add(d.canonical_name.lower())
            expanded.add(d.canonical_id.lower())
            for syn in d.synonyms:
                if len(syn) > 2:
                    expanded.add(syn.lower())

        return expanded

    def get_all_entries(self) -> Dict[str, TermMatch]:
        """Return the dictionary entries."""
        return dict(self._entries)
