import os, json, streamlit as st, requests
st.set_page_config(page_title="AI Public Cloud Advisor", page_icon="☁️")
st.title("☁️ AI Public Cloud Advisor")
st.caption("Azure vs AWS recommendation with RAG + Multi-Agent (LangChain + LangGraph)")

with st.sidebar:
    st.header("Parameters")
    api_url = st.text_input("API URL", value=os.environ.get("API_URL","http://localhost:8000/advise"))
    monthly_users = st.number_input("Monthly Users", value=100000, step=1000)
    region = st.selectbox("Primary Region", ["apac","us","eu"])
    storage_gb = st.number_input("Storage (GB)", value=200, step=50)
    compute_tier = st.selectbox("Compute Tier", ["small","medium","large"])
    st.divider()
    industry = st.selectbox("업종 (Industry)", ["general","fintech","ecommerce","healthcare","media","gaming","manufacturing","public"])
    compliance = st.text_input("컴플라이언스 요구사항", placeholder="예) PCI-DSS, HIPAA, ISO27001, K-ISMS")
    existing_cloud = st.selectbox("현재 사용 중인 클라우드", ["none","AWS","Azure","GCP","On-premise","Hybrid"])

st.write("### Describe your project requirements")
user_prompt = st.text_area("Prompt", height=160, placeholder="예) 전자상거래 웹앱, 한국/일본 트래픽, 카드결제, PII 암호화, 월 예산 5천 달러...")

if st.button("Get Recommendation", type="primary"):
    with st.spinner("Thinking..."):
        payload = {
            "prompt": user_prompt,
            "monthly_users": int(monthly_users),
            "region": region,
            "storage_gb": int(storage_gb),
            "compute_tier": compute_tier,
            "industry": industry,
            "compliance": compliance or None,
            "existing_cloud": None if existing_cloud == "none" else existing_cloud,
        }
        try:
            r = requests.post(api_url, json=payload, timeout=120)
            r.raise_for_status()
            result = r.json().get("result","{}")
            try:
                data = json.loads(result)
                st.success(f"Recommended Provider: **{data.get('provider','N/A')}**")
                st.subheader("Rationale"); st.write(data.get("rationale",""))
                st.subheader("Proposed Architecture"); st.write(data.get("architecture",""))
                st.subheader("Estimated Monthly Cost (USD)"); st.write(data.get("monthly_cost_estimate",""))
                st.subheader("Next Steps"); st.write(data.get("next_steps",""))
            except Exception:
                st.warning("Model did not return strict JSON; showing raw result below.")
                st.code(result, language="json")
        except Exception as e:
            st.error(f"Request failed: {e}")
