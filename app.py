import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

model = open("Sequential.pkl", "rb")
rf = joblib.load(model)
st.title("Прогнозування серцево-судинних хвороб")

YES_NO = ['Так', 'Ні']
sex = st.selectbox("Стать", options=("Чоловіча", "Жіноча"))
age = st.number_input("Вік", 0, 90, 30)
height = st.slider('Зріст (см)', value=180.0, min_value=100.0, max_value=220.0, step=1.0, format="%.1f")
weight = st.slider('Вага (кг)', value=75.0, min_value=0.0, max_value=150.0, step=1.0, format="%.1f")
ChSS = st.number_input("Пульс (в стані спокою)", 0, 120, 70)
ADsist = st.number_input("Систолічний тиск", 90, 160, 120)
ADdiast = st.number_input("Діастолічний тиск", 50, 120, 80)

cholesterin = st.selectbox("Показник холестерину: ",
                           options=("Норма - завжди в межах норми",
                                    "Вище в межах норми - від 4,7 ммоль/л до 5",
                                    "Іноді вище норми",
                                    "Завжди вище норми"))

diabetus = st.selectbox("Ви хворієте на діабет?", options=("Ні", "Так"))

OP = st.selectbox("Чи були оперативні втручання стосовно серцево-судинної системи?", options=("Ні", "Так"))

Shunt = st.selectbox("Чи робили Вам шунтування артерій?", options=("Ні", "Так"))

AGtherapia = st.selectbox("Чи проходили ви антигіпертензивну терапію?", options=("Ні", "Так"))

IMT = round((weight / height ** 2), 2)
data = {
    "OP": [OP],
    "Shunt": [Shunt],
    "age": [age],
    "height": [height],
    "weight": [weight],
    "IMT": [IMT],
    "sex": [sex],
    "ChSS": [ChSS],
    "AD sist.": [ADsist],
    "AD diast": [ADdiast],
    "AG therapia": [AGtherapia],
    "cholesterin": [cholesterin],
    "diabetus melitus": [diabetus],
}
input_df = pd.DataFrame(data, index=[0])
label = LabelEncoder()
for col in input_df.columns.values.tolist():
    if pd.api.types.is_string_dtype(input_df[col].dtype):
        input_df[col] = label.fit_transform(input_df[col])
        le_name_mapping = dict(zip(label.classes_, label.transform(label.classes_)))
        # print(le_name_mapping)
st.divider()


submit = st.button("Передбачити")
if submit:
    print(input_df)
    prediction = rf.predict(input_df)
    # prediction_prob = rf.predict_proba(input_df)
    st.markdown(f"**Вірогідність наявності серцево-судинної хвороби становить {round(prediction[0][0] * 100, 2)}%**")

