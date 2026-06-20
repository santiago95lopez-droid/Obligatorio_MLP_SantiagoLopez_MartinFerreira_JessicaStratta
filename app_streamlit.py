import requests
import streamlit as st

API_BASE_URL = "http://localhost:8080"
IMAGE_URL = f"{API_BASE_URL}/classification/images"
BATCH_URL = f"{API_BASE_URL}/classification/predict-batch"

st.title("Clasificador de Hongos ORT")
uploaded_file = st.file_uploader("Selecciona una imagen o ZIP", type=["png", "jpg", "jpeg", "zip"])

if uploaded_file:
    filename = uploaded_file.name.lower()
    is_zip = filename.endswith(".zip")
    is_image = filename.endswith(('.png', '.jpg', '.jpeg'))

    if is_image:
        st.image(uploaded_file, caption="Imagen cargada", use_column_width=True)

    if st.button("Clasificar"):
        with st.spinner("Clasificando..."):
            try:
                if is_zip:
                    st.info("Archivo Batch ZIP detectado.")
                    files = {"archive": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(BATCH_URL, files=files, timeout=60)
                elif is_image:
                    files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(IMAGE_URL, files=files, timeout=30)
                else:
                    st.error("Formato no soportado. Usa PNG, JPG, JPEG o ZIP.")
                    response = None

                if response is None:
                    st.stop()

                response.raise_for_status()

                if is_image:
                    try:
                        data = response.json()
                    except ValueError:
                        label = response.text.strip()
                        st.warning(label)
                        st.stop()

                    if isinstance(data, str):
                        st.warning(data)
                        st.stop()

                    if not isinstance(data, dict):
                        st.error("Respuesta inválida del servidor.")
                        st.stop()

                    image_info = data.get("images", [{}])[0]
                    label = image_info.get("label", "Desconocido")
                    score = image_info.get("score", 0.0) * 100
                    st.metric("Predicción", label, f"{score:.1f}%")
                    st.success(f"Resultado: {label} (Confianza: {score:.1f}%)")
                else:
                    try:
                        data = response.json()
                    except ValueError:
                        st.warning(response.text.strip())
                        st.stop()

                    predictions = data.get("predictions", [])
                    if not predictions:
                        st.error("No se devolvieron predicciones para el batch.")
                    else:
                        table = []
                        for item in predictions:
                            score = item.get("score", item.get("confidence"))
                            if isinstance(score, (int, float)):
                                score = f"{score * 100:.1f}%"
                            table.append({
                                "archivo": item.get("filename", "-"),
                                "predicción": item.get("prediction", "-"),
                                "confianza": score or "N/A",
                            })
                        st.dataframe(table)
            except requests.exceptions.RequestException as exc:
                st.error(f"No se pudo conectar con la API: {exc}")
            except ValueError:
                st.error("Respuesta inválida del servidor.")
