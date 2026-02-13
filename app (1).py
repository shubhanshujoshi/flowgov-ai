
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
feature_columns = joblib.load("features.pkl")

st.set_page_config(page_title="FlowGov AI Advanced", layout="wide")

departments = ["General", "Ortho", "ENT"]

# ---------------- SESSION STATE ----------------
if "queues" not in st.session_state:
    st.session_state.queues = {dept: [] for dept in departments}

if "on_hold" not in st.session_state:
    st.session_state.on_hold = {dept: [] for dept in departments}

if "doctor_count" not in st.session_state:
    st.session_state.doctor_count = {dept: 2 for dept in departments}

# 🔥 Department-wise token counters
if "token_counter" not in st.session_state:
    st.session_state.token_counter = {dept: 1 for dept in departments}

if "arrival_log" not in st.session_state:
    st.session_state.arrival_log = []

if "lunch" not in st.session_state:
    st.session_state.lunch = False

hall_capacity = 40


# ---------------- PREDICT FUNCTION ----------------
def predict_wait(position, staff, dept, priority):

    input_dict = {
        "day": 1,
        "arrival_time_min": 60,
        "hour_of_day": 11,
        "is_lunch_period": 1 if st.session_state.lunch else 0,
        "is_rush_hour": 0,
        "queue_length_at_arrival": position,
        "base_service_time": 12,
        "real_time_velocity": 12,
        "staff_on_duty": staff,
        "is_priority": 1 if priority else 0,
        "is_no_show": 0,
        "service_type_General": 1 if dept=="General" else 0,
        "service_type_Ortho": 1 if dept=="Ortho" else 0,
        "service_type_ENT": 1 if dept=="ENT" else 0
    }

    df_input = pd.DataFrame([input_dict])
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)

    return round(model.predict(df_input)[0], 2)


# ---------------- TITLE ----------------
st.title("🏥 FlowGov AI – Advanced Smart Queue System")

tabs = st.tabs([
    "👨‍⚕️ Admin Dashboard",
    "📱 Patient View",
    "📊 Analytics"
])

# ======================================================
# ADMIN DASHBOARD
# ======================================================
with tabs[0]:

    st.subheader("Doctor Availability & Break Scheduling")

    for dept in departments:
        col1, col2 = st.columns(2)

        available = col1.number_input(
            f"{dept} Doctors Available",
            1, 10,
            st.session_state.doctor_count[dept],
            key=f"doc_{dept}"
        )

        break_count = col2.number_input(
            f"{dept} Doctors On Break",
            0, available,
            0,
            key=f"break_{dept}"
        )

        # Effective doctors
        st.session_state.doctor_count[dept] = max(1, available - break_count)

    st.session_state.lunch = st.checkbox("🍽 Global Lunch Mode")

    # ---------------- ADD PATIENT ----------------
    st.subheader("Add Patient")

    dept_select = st.selectbox("Department", departments)
    emergency = st.checkbox("Emergency Case")

    if st.button("➕ Add Patient"):

        token_number = st.session_state.token_counter[dept_select]
        prefix = dept_select[0]   # G, O, E

        token = f"{prefix}{token_number}"

        st.session_state.token_counter[dept_select] += 1

        patient = {
            "id": token,
            "priority": emergency
        }

        if emergency:
            st.session_state.queues[dept_select].insert(0, patient)
        else:
            st.session_state.queues[dept_select].append(patient)

        st.session_state.arrival_log.append(dept_select)

    # ---------------- ZIPPER LOGIC ----------------
    st.subheader("Missed Patients (Zipper Logic)")

    for dept in departments:

        st.markdown(f"### {dept} Queue")

        queue = st.session_state.queues[dept]
        hold = st.session_state.on_hold[dept]

        if queue:
            df_queue = pd.DataFrame(queue)
            df_queue.insert(0, "Position", range(1, len(queue)+1))
            st.dataframe(df_queue, use_container_width=True)

            if st.button(f"▶️ Serve Next {dept}", key=f"serve_{dept}"):
                missed = queue.pop(0)
                st.session_state.on_hold[dept].append(missed)
        else:
            st.info("No patients waiting")

        if hold:
            st.markdown("On Hold:")
            st.write(hold)

            if st.button(f"🔄 Reinsert (Zipper) {dept}", key=f"zip_{dept}"):
                patient = hold.pop(0)
                insert_position = min(2, len(queue))
                queue.insert(insert_position, patient)

    # ---------------- SURGE RADAR ----------------
    st.subheader("📡 Surge Prediction Radar")

    total_queue = sum(len(st.session_state.queues[d]) for d in departments)
    predicted_surge = total_queue + np.random.randint(0, 5)

    st.metric("Current Queue", total_queue)
    st.metric("Predicted Next 30 min", predicted_surge)

    if predicted_surge > 20:
        st.error("🔴 Surge Expected – Open More Counters")
    elif predicted_surge > 10:
        st.warning("🟡 Moderate Surge")
    else:
        st.success("🟢 Stable Flow")


# ======================================================
# PATIENT VIEW
# ======================================================
with tabs[1]:

    st.subheader("Check Token Status")

    token_input = st.text_input("Enter Token (e.g., G1, O2, E3)")

    if token_input:

        token_input = token_input.strip().upper()

        found = False

        for dept in departments:

            queue = st.session_state.queues[dept]

            for i, patient in enumerate(queue):

                if patient["id"] == token_input:

                    staff = st.session_state.doctor_count[dept]

                    wait = predict_wait(
                        i,
                        staff,
                        dept,
                        patient["priority"]
                    )

                    st.success(f"Department: {dept}")
                    st.write("People Ahead:", i)
                    st.write("Estimated Wait:", wait, "minutes")

                    found = True
                    break

        if not found:
            st.error("Token not found")


# ======================================================
# ANALYTICS DASHBOARD
# ======================================================
with tabs[2]:

    st.subheader("Department Demand Analytics")

    if st.session_state.arrival_log:

        df = pd.DataFrame(st.session_state.arrival_log, columns=["Department"])
        counts = df["Department"].value_counts()

        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax)
        ax.set_title("Department Demand Distribution")
        ax.set_ylabel("Number of Patients")

        st.pyplot(fig)

    else:
        st.info("No arrival data yet.")
