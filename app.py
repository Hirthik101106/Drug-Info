import streamlit as st
from pubchempy import get_compounds
from chembl_webresource_client.new_client import new_client
import pandas as pd
import spacy
from transformers import pipeline
import socket
import logging
from datetime import datetime

# --- Configuration ---
st.set_page_config(
    page_title="Drug-Info",
    page_icon="🧪", 
    layout="wide"
)

# Initialize with empty session state
if 'run_count' not in st.session_state:
    st.session_state.run_count = 0
    st.session_state.clear()

# Minimal logging (no storage)
logging.basicConfig(level=logging.WARNING)

def get_biomedical_data(query):
    """Fetch fresh data without storage"""
    data = {
        'cid': None,
        'name': query,
        'bioactivities': []
    }
    
    try:
        # PubChem (no caching)
        compounds = get_compounds(query, namespace='name', timeout=10)
        if not compounds:
            return None
            
        compound = compounds[0]
        data.update({
            'cid': compound.cid,
            'name': compound.iupac_name or query,
            'formula': getattr(compound, 'molecular_formula', None),
            'smiles': getattr(compound, 'canonical_smiles', None)
        })

        # ChEMBL (fresh connection each time)
        chembl = new_client
        chembl.timeout = 15
        
        if data.get('smiles'):
            results = list(chembl.molecule.filter(
                molecule_structures__canonical_smiles=data['smiles']
            ))
            if results:
                activities = chembl.activity.filter(
                    molecule_chembl_id=results[0]['molecule_chembl_id'],
                    standard_type__in=["IC50", "Ki"],
                    standard_units="nM"
                )[:10]  # Limit results
                
                for act in activities:
                    data['bioactivities'].append({
                        'target': act.get('target_chembl_id'),
                        'value': act.get('standard_value'),
                        'type': act.get('standard_type')
                    })

    except Exception as e:
        logging.warning(f"Data fetch error: {str(e)}")
        return None
    
    return data

def main():
    st.title("Drug-Info")
    
    # Input resets on full refresh but maintains during interaction
    query = st.text_input("Enter compound:", 
                         value=st.session_state.get('last_query', ''))
    
    if st.button("Analyze"):
        st.session_state.run_count += 1
        st.session_state.last_query = query
        
        with st.spinner("Fetching fresh data..."):
            data = get_biomedical_data(query)
            
            if data:
                # Display (no storage)
                st.subheader("Chemical Properties")
                cols = st.columns(2)
                cols[0].metric("Name", data['name'])
                cols[1].metric("Formula", data['formula'] or "Unknown")
                
                if data['bioactivities']:
                    st.subheader("Bioactivities")
                    st.dataframe(
                        pd.DataFrame(data['bioactivities']),
                        use_container_width=True,
                        hide_index=True
                    )

if __name__ == "__main__":
    main()
