import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Food Recommendation System", page_icon="🍔", layout="wide"
)

st.title("🍔 Food Recommendation System")
st.write("Food Recommendation System using Apriori Algorithm and Association Rules")

rules = pd.read_csv("association_rules.csv")
baskets = pd.read_csv("processed_baskets.csv")

st.success("Dataset and association rules loaded successfully!")

st.metric("Total Transactions", len(baskets))
st.metric("Total Association Rules", len(rules))

st.divider()
st.header("Food Recommendations")

# Get all available food items
food_items = sorted(baskets.columns.tolist())

# Let the user select multiple foods
selected_foods = st.multiselect(
    "Select food items from your basket:",
    options=food_items,
    placeholder="Choose your food...",
)


# Recommendation function
def recommend_food(selected_items, rules_df, top_n=5):
    selected_items = set(selected_items)
    recommendations = []

    for _, rule in rules_df.iterrows():
        antecedent = set(rule["antecedents"].split(", "))
        consequent = set(rule["consequents"].split(", "))

        if antecedent.issubset(selected_items):
            for item in consequent - selected_items:
                if rule["lift"] > 1:
                    recommendations.append(
                        {
                            "Recommended Item": item,
                            "Based On": rule["antecedents"],
                            "Support": rule["support"],
                            "Confidence": rule["confidence"],
                            "Lift": rule["lift"],
                        }
                    )

    if not recommendations:
        return pd.DataFrame()

    result = pd.DataFrame(recommendations)

    return (
        result.sort_values(["Lift", "Confidence"], ascending=False)
        .drop_duplicates(subset=["Recommended Item"])
        .head(top_n)
        .reset_index(drop=True)
    )


if selected_foods:
    recommendations = recommend_food(selected_foods, rules, top_n=5)

    if not recommendations.empty:
        st.subheader("Recommended for You")
        st.dataframe(
            recommendations.round(4), use_container_width=True, hide_index=True
        )
    else:
        st.info("No strong recommendations found for this basket.")
else:
    st.info("Select at least one food item to see recommendations.")

st.divider()
st.header("Food Popularity Analysis")

# Count how many orders contain each food
food_counts = baskets.sum().sort_values(ascending=False)

top_10_foods = food_counts.head(10).reset_index()
top_10_foods.columns = ["Food Item", "Order Count"]

fig = px.bar(
    top_10_foods,
    x="Food Item",
    y="Order Count",
    title="Top 10 Most Frequently Ordered Foods",
    text="Order Count",
)

fig.update_layout(xaxis_tickangle=-45)

st.plotly_chart(fig, use_container_width=True)

st.divider()
st.header("Association Rule Threshold Playground")

st.write(
    "Adjust the thresholds to explore how they affect "
    "the number of association rules."
)

col1, col2 = st.columns(2)

with col1:
    min_support = st.slider(
        "Minimum Support", min_value=0.01, max_value=0.05, value=0.01, step=0.005
    )

with col2:
    min_confidence = st.slider(
        "Minimum Confidence", min_value=0.10, max_value=0.25, value=0.10, step=0.01
    )

# Filter existing rules
filtered_rules = rules[
    (rules["support"] >= min_support) & (rules["confidence"] >= min_confidence)
]

filtered_positive_rules = filtered_rules[filtered_rules["lift"] > 1]

col3, col4 = st.columns(2)

with col3:
    st.metric("Rules Matching Thresholds", len(filtered_rules))

with col4:
    st.metric("Positive Association Rules", len(filtered_positive_rules))

st.subheader("Filtered Association Rules")

st.dataframe(
    filtered_positive_rules[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ].round(4),
    use_container_width=True,
    hide_index=True,
)
