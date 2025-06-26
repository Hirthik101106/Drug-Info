import streamlit as st
from pubchempy import get_compounds
from chembl_webresource_client.new_client import new_client
import pandas as pd
import spacy
from tenacity import retry, stop_after_attempt
import socket
import logging

# --- Configuration ---
st.set_page_config(page_title="CureGenie Pro 🧬", page_icon="🧪", layout="wide")
logging.basicConfig(filename='curegenie.log', level=logging.INFO, filemode='w')

def check_internet():
    try:
        socket.create_connection(("www.google.com", 80), timeout=5)
        return True
    except OSError:
        return False

if not check_internet():
    st.error("⚠ No internet connection. Some features may be limited.")
    st.stop()

@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except Exception as e:
        st.warning(f"⚠ NLP model not available: {e}")
        return None

def safe_float_format(value, decimal_places=2):
    try:
        return f"{float(value):.{decimal_places}f}" if value is not None else 'N/A'
    except:
        return 'N/A'

@retry(stop=stop_after_attempt(3))
@st.cache_data(ttl=3600, max_entries=3)  # ONLY CHANGE: max_entries=20 → max_entries=3
def get_biomedical_data(query, input_type="name"):
    data = {
        'cid': None, 
        'name': query, 
        'targets': [], 
        'proteins': [], 
        'bioactivities': [],
        'debug_log': []
    }
    
    # --- PubChem Fetch ---
    try:
        compounds = get_compounds(query, namespace=input_type, fields=["cid", "iupac_name", "molecular_formula", "molecular_weight", "canonical_smiles", "inchikey"])
        if not compounds:
            st.warning(f"No PubChem match for: {query}")
            return None
        compound = compounds[0]
        data.update({
            'cid': compound.cid,
            'name': compound.iupac_name or query,
            'formula': compound.molecular_formula,
            'weight': float(compound.molecular_weight) if compound.molecular_weight else None,
            'smiles': compound.canonical_smiles,
            'inchi_key': compound.inchikey
        })
    except Exception as e:
        data['debug_log'].append(f"PubChem error: {e}")

    # --- ChEMBL Fetch ---
    try:
        lookup_methods = []
        if input_type == "name":
            lookup_methods.append(("molecule_synonyms__molecule_synonym__iexact", query))
        if data.get('inchi_key'):
            lookup_methods.append(("molecule_structures__standard_inchi_key", data['inchi_key']))
        if data.get('smiles'):
            lookup_methods.append(("molecule_structures__canonical_smiles", data['smiles']))

        chembl_id = None
        for method, value in lookup_methods:
            try:
                results = list(new_client.molecule.filter(**{method: value}).only("molecule_chembl_id"))
                if results:
                    chembl_id = results[0]['molecule_chembl_id']
                    break
            except Exception as e:
                data['debug_log'].append(f"ChEMBL lookup failed via {method}: {e}")

        if chembl_id:
            activities = new_client.activity.filter(
                molecule_chembl_id=chembl_id,
                standard_type__in=["IC50", "Ki", "Kd", "EC50", "AC50", "Potency"],
                standard_relation="=",
                standard_units__in=["nM", "μM"]
            ).only([
                'target_chembl_id', 'standard_value', 'standard_units',
                'standard_type', 'assay_description', 'pchembl_value'
            ])[:15]

            for act in activities:
                try:
                    target = new_client.target.get(act['target_chembl_id'])
                    data['bioactivities'].append({
                        'target': target['pref_name'],
                        'value': safe_float_format(act.get('standard_value')),
                        'unit': act.get('standard_units'),
                        'pIC50': safe_float_format(act.get('pchembl_value')),
                        'evidence': act.get('assay_description', 'No description'),
                        'source': 'ChEMBL'
                    })
                    if act['target_chembl_id'] not in data['targets']:
                        data['targets'].append(act['target_chembl_id'])
                except Exception as e:
                    data['debug_log'].append(f"Activity processing error: {e}")
    except Exception as e:
        data['debug_log'].append(f"ChEMBL API failed: {e}")

    # --- Protein Targets ---
    try:
        for target_id in data['targets'][:3]:
            try:
                target_info = new_client.target.get(target_id)
                if target_info.get('target_components'):
                    acc = target_info['target_components'][0].get('accession')
                    if acc:
                        data['proteins'].append({
                            'accession': acc,
                            'name': target_info['pref_name'],
                            'source': "UniProt"
                        })
            except Exception as e:
                data['debug_log'].append(f"UniProt fetch failed for {target_id}: {e}")
    except Exception as e:
        data['debug_log'].append(f"UniProt block error: {e}")

    return data

def main():
    nlp = load_models()
    st.title("🧬 CureGenie Pro (Bioactivity Fixed)")

    col1, col2 = st.columns([3, 1])
    with col1:
        input_type = st.selectbox("Input Type", ["Drug Name", "SMILES", "InChI Key"])
        query = st.text_input("Enter compound:", placeholder="e.g., aspirin", value="aspirin")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("Analyze", type="primary")

    if analyze_btn and query.strip():
        with st.spinner("Fetching data..."):
            input_map = {
                "Drug Name": "name",
                "SMILES": "smiles",
                "InChI Key": "inchikey"
            }
            data = get_biomedical_data(query.strip(), input_map[input_type])

            if data:
                st.success("✅ Data retrieved successfully")
                
                st.subheader("🧪 Chemical Properties")
                chem_props = pd.DataFrame({
                    "Property": ["CID", "Name", "Formula", "Weight", "SMILES", "InChI Key"],
                    "Value": [
                        data.get('cid', 'N/A'),
                        data.get('name', 'N/A'),
                        data.get('formula', 'N/A'),
                        safe_float_format(data.get('weight')),
                        data.get('smiles', 'N/A'),
                        data.get('inchi_key', 'N/A')
                    ]
                })
                st.dataframe(chem_props)

                st.subheader("📊 Bioactivities (Top 15)")
                if data.get('bioactivities'):
                    bio_df = pd.DataFrame(data['bioactivities'])
                    st.dataframe(bio_df[['target', 'value', 'unit', 'pIC50', 'evidence', 'source']])
                else:
                    st.warning("No bioactivity data found. Try another compound.")

                with st.expander("🐛 Debug Log", expanded=False):
                    st.code("\n".join(data.get("debug_log", [])))
            else:
                st.error("❌ No data found for this compound")

    st.caption("⚡ Powered by PubChem + ChEMBL | Fixed bioactivity display")

if __name__ == "__main__":
    main()
