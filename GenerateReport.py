import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from sklearn.metrics import accuracy_score, classification_report


class ReportGenerator:
    # -----------------------------------------------------------------------------------------------------------------#
    def __init__(self, data_processor, model_builder, output_pdf="report.pdf"):
        self.data_processor = data_processor
        self.model_builder = model_builder
        self.output_pdf = output_pdf
        self.styles = getSampleStyleSheet()

    # -----------------------------------------------------------------------------------------------------------------#
    def generate_report(self):
        doc = SimpleDocTemplate(self.output_pdf, pagesize=letter,
                                rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        elements = []

        title_style = self.styles["Title"]
        elements.append(Paragraph("Customer Churn Prediction Report", title_style))
        elements.append(Spacer(1, 20))

        intro_text = "This report summarizes the preprocessing, model evaluation, and findings from the customer churn prediction project."
        elements.append(Paragraph(intro_text, self.styles["BodyText"]))
        elements.append(Spacer(1, 20))

        elements.extend(self.add_data_summary())
        elements.extend(self.add_eda_section())
        elements.extend(self.add_model_evaluation())
        elements.extend(self.add_conclusion())

        doc.build(elements)
        print(f"Report saved to {self.output_pdf}")

    # -----------------------------------------------------------------------------------------------------------------#
    def add_data_summary(self):
        content = []
        content.append(Paragraph("Data Summary:", self.styles["Heading2"]))
        content.append(Spacer(1, 12))

        summary_text = [
            f"Number of data points: {len(self.data_processor.customer_data)}",
            "Missing values handled and categorical data encoded.",
            "Features scaled for modeling.",
        ]
        for line in summary_text:
            content.append(Paragraph(line, self.styles["BodyText"]))
            content.append(Spacer(1, 6))

        return content

    # -----------------------------------------------------------------------------------------------------------------#
    def add_eda_section(self):
        content = []
        content.append(Paragraph("Exploratory Data Analysis (EDA):", self.styles["Heading2"]))
        content.append(Spacer(1, 12))
        content.append(Paragraph("The following features were visualized using histograms:", self.styles["BodyText"]))
        content.append(Spacer(1, 6))

        for feature in self.data_processor.numerical_features:
            content.append(Paragraph(f"- {feature}", self.styles["BodyText"]))
            content.append(Spacer(1, 6))

        image_path = "tenure_histogram.png"
        if os.path.exists(image_path):
            from reportlab.platypus import Image
            img = Image(image_path, width=400, height=200)
            content.append(Spacer(1, 12))
            content.append(img)
        else:
            content.append(Paragraph("Image not found: tenure_histogram.png", self.styles["BodyText"]))

        return content

    # -----------------------------------------------------------------------------------------------------------------#
    def add_model_evaluation(self):
        content = []

        content.append(Paragraph("Model Evaluation:", self.styles["Heading2"]))
        content.append(Spacer(1, 12))

        for model_name in self.model_builder.models:
            model = self.model_builder.models[model_name]

            if not hasattr(model, "coef_") and not hasattr(model, "feature_importances_"):
                model.fit(self.data_processor.X_train, self.data_processor.y_train)

            accuracy = accuracy_score(self.data_processor.y_test, model.predict(self.data_processor.X_test))

            content.append(Paragraph(f"Model: {model_name}", self.styles["Heading3"]))
            content.append(Spacer(1, 6))
            content.append(Paragraph(f"Accuracy: {accuracy:.2f}", self.styles["BodyText"]))
            content.append(Spacer(1, 6))

            report = classification_report(self.data_processor.y_test, model.predict(self.data_processor.X_test), output_dict=True)
            table_data = [["Metric", "Precision", "Recall", "F1-Score", "Support"]]
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    table_data.append([label,
                                       f"{metrics.get('precision', 'N/A'):.2f}",
                                       f"{metrics.get('recall', 'N/A'):.2f}",
                                       f"{metrics.get('f1-score', 'N/A'):.2f}",
                                       f"{metrics.get('support', 'N/A'):.0f}"])
                else:
                    continue

            table = Table(table_data, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]))
            content.append(table)
            content.append(Spacer(1, 12))

        return content

    # -----------------------------------------------------------------------------------------------------------------#
    def add_conclusion(self):
        content = []

        content.append(Paragraph("Conclusion:", self.styles["Heading2"]))
        content.append(Spacer(1, 12))
        conclusion_text = "The model performance has been evaluated for several classifiers, including Random Forest, Logistic Regression, and Decision Tree. The best-performing model based on accuracy and other metrics will be used for predicting customer churn."
        content.append(Paragraph(conclusion_text, self.styles["BodyText"]))

        return content
