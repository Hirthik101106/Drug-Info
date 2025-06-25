import streamlit as st
from pubchempy import get_compounds
from chembl_webresource_client.new_client import new_client
import requests
import pandas as pd
from st_aggrid import AgGrid
import spacy
from tenacity import retry, stop_after_attempt
from transformers import pipeline
import socket
import logging

# --- Configuration ---
st.set_page_config(page_title="CureGenie Pro 🧬", page_icon="🧪", layout="wide")
logging.basicConfig(filename='curegenie.log', level=logging.INFO)

def check_internet():
    try:
        socket.create_connection(("www.google.com", 80), timeout=5)
        return True
    except OSError:
        return False

if not check_internet():
    st.error("⚠️ No internet connection. Some features may be limited.")
    st.stop()

@st.cache_resource
def load_models():
    models = {}
    try:
        try:
            models['ner'] = spacy.load("en_core_sci_sm")
            st.success("✅ Loaded biomedical NLP model (en_core_sci_sm)")
        except:
            models['ner'] = spacy.load("en_core_web_sm")
            st.warning("⚠️ Falling back to general NLP model (en_core_web_sm)")
        models['summarizer'] = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        models['ner'] = None
        models['summarizer'] = None
    return models

def safe_float_format(value, decimal_places=2):
    try:
        return f"{float(value):.{decimal_places}f}" if value is not None else 'N/A'
    except:
        return 'N/A'

@retry(stop=stop_after_attempt(3))
@st.cache_data(ttl=3600)
def get_biomedical_data(query, input_type="name"):
    data = {'cid': None, 'name': query, 'targets': [], 'proteins': [], 'bioactivities': [], 'debug_log': []}
    try:
        compounds = get_compounds(query, namespace=input_type)
        if not compounds:
            st.warning(f"No PubChem match for: {query}")
            return None
        compound = compounds[0]
        data.update({
            'cid': compound.cid,
            'name': compound.iupac_name or query,
            'formula': getattr(compound, 'molecular_formula', None),
            'weight': float(compound.molecular_weight) if compound.molecular_weight else None,
            'smiles': getattr(compound, 'canonical_smiles', None),
            'inchi_key': getattr(compound, 'inchi_key', None)
        })
    except Exception as e:
        data['debug_log'].append(f"PubChem error: {e}")

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
                    data['debug_log'].append(f"Found ChEMBL ID via {method}: {chembl_id}")
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
            ])[:10]

            for act in activities:
                try:
                    target = new_client.target.get(act['target_chembl_id'])
                    data['bioactivities'].append({
                        'target': target['pref_name'],
                        'target_id': act['target_chembl_id'],
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

def render_chemical_data(data):
    return pd.DataFrame({
        "Property": ["CID", "Name", "Formula", "Weight", "SMILES", "InChI Key"],
        "Value": [
            str(data.get('cid', 'N/A')),
            data.get('name', 'N/A'),
            data.get('formula', 'N/A'),
            safe_float_format(data.get('weight')),
            data.get('smiles', 'N/A')[:50] + '...' if data.get('smiles') else 'N/A',
            data.get('inchi_key', 'N/A')
        ]
    })

def render_target_data(data):
    if not data.get('proteins'):
        return pd.DataFrame({"Status": ["No targets found"], "Debug Info": ["\n".join(data.get('debug_log', []))]})
    return pd.DataFrame(data['proteins'])

def render_bioactivities(data):
    if not data.get('bioactivities'):
        return pd.DataFrame({"Note": ["No bioactivity data found"]})
    df = pd.DataFrame(data['bioactivities'])
    return df.sort_values("pIC50", ascending=False)[['target', 'value', 'unit', 'pIC50', 'evidence', 'source']]

# ✅ Improved Summary Function (only this changed)
def generate_safe_summary(data, summarizer, max_length=400):
    if not summarizer:
        return "Summary unavailable"

    try:
        compound_name = data.get('name', 'Unknown')
        formula = data.get('formula', 'N/A')
        weight = safe_float_format(data.get('weight'))
        protein_targets = [p['name'] for p in data.get('proteins', []) if p.get('name')]
        target_list = ', '.join(protein_targets) if protein_targets else 'No protein targets identified.'

        bio_lines = []
        for act in data.get('bioactivities', [])[:5]:
            target = act.get('target')
            if not target or len(target) > 100 or all(c.isdigit() for c in target.strip()):
                continue
            value = safe_float_format(act.get('value'))
            unit = act.get('unit', 'N/A')
            pIC50 = safe_float_format(act.get('pIC50'))
            bio_lines.append(f"{target} (pIC50: {pIC50}, {value} {unit})")

        bio_summary = "; ".join(bio_lines) if bio_lines else "No significant bioactivity data found."

        input_text = (
            f"{compound_name} is a compound with the molecular formula {formula} and a molecular weight of {weight} g/mol. "
            f"Identified protein targets include: {target_list}. "
            f"Bioactivity profile: {bio_summary}"
        )

        truncated = input_text.strip()[:3000]
        summary = summarizer(truncated, max_length=max_length, min_length=100)
        return summary[0]['summary_text']
    except Exception as e:
        logging.warning(f"Summarization failed: {str(e)}")
        return "Summary unavailable"

# --- Main UI ---
def main():
    models = load_models()
    st.title("🧬 CureGenie Pro")

    col1, col2 = st.columns([3, 1])
    with col1:
        input_type = st.selectbox("Input Type", ["Drug Name", "SMILES", "InChI Key"])
        drug_name = st.text_input("Enter compound identifier:", placeholder="e.g., aspirin, ibuprofen")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("Analyze", type="primary")

    st.caption("🔍 Examples: aspirin, atorvastatin, caffeine")

    if analyze_btn:
        if not drug_name.strip():
            st.warning("Please enter a compound name or identifier")
        else:
            with st.spinner("Fetching biomedical data..."):
                input_map = {
                    "Drug Name": "name",
                    "SMILES": "smiles",
                    "InChI Key": "inchikey"
                }
                data = get_biomedical_data(drug_name.strip(), input_map[input_type])
                if data:
                    st.success("✅ Data retrieved successfully")
                    with st.expander("🐛 Debug Information", expanded=False):
                        st.code("\n".join(data.get("debug_log", [])))

                    st.subheader("🧪 Chemical Properties")
                    try:
                        AgGrid(render_chemical_data(data), height=150)
                    except:
                        st.dataframe(render_chemical_data(data))

                    st.subheader("🎯 Biological Targets")
                    st.dataframe(render_target_data(data))

                    st.subheader("📊 Bioactivities")
                    st.dataframe(render_bioactivities(data))

                    if models.get('ner') or models.get('summarizer'):
                        st.subheader("📝 NLP Insights")
                        if models.get('ner'):
                            text = f"{data.get('name')} targets {len(data.get('proteins'))} proteins"
                            doc = models['ner'](text)
                            ents = [(ent.text, ent.label_) for ent in doc.ents]
                            if ents:
                                st.table(pd.DataFrame(ents, columns=["Entity", "Type"]))
                        if models.get('summarizer'):
                            st.write("**Summary:**", generate_safe_summary(data, models['summarizer']))
                else:
                    st.error("❌ No data retrieved")

    st.divider()
    st.caption("⚡ Powered by PubChem, ChEMBL, UniProt")

if __name__ == "__main__":
    main()

