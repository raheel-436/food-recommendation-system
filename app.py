import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go

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

st.divider()
st.header("Association Rules Visualization")

if not filtered_positive_rules.empty:

    top_rules = (
        filtered_positive_rules.sort_values("lift", ascending=False).head(10).copy()
    )

    top_rules["Rule"] = top_rules["antecedents"] + " → " + top_rules["consequents"]

    fig = px.bar(
        top_rules,
        x="lift",
        y="Rule",
        orientation="h",
        title="Top 10 Association Rules by Lift",
        text="lift",
        color="confidence",
        color_continuous_scale="Blues",
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Lift",
        yaxis_title="Association Rule",
    )

    fig.add_vline(x=1, line_dash="dash", line_color="red")

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No positive association rules found " "for the selected thresholds.")

st.divider()
st.header("🛒 Smart Basket Analysis")

smart_basket = st.multiselect(
    "Build your food basket:", options=food_items, key="smart_basket"
)

if smart_basket:
    st.write("### Your Basket")
    st.write(", ".join(smart_basket))

    # Find recommendations
    smart_recommendations = recommend_food(smart_basket, rules, top_n=5)

    if not smart_recommendations.empty:
        st.subheader("Suggested Additions")

        for _, rec in smart_recommendations.iterrows():
            with st.container(border=True):
                st.write(f"**{rec['Recommended Item']}**")

                st.write(
                    f"Recommended because your basket contains "
                    f"**{rec['Based On']}**."
                )

                st.write(
                    f"Confidence: {rec['Confidence']:.2%} | " f"Lift: {rec['Lift']:.2f}"
                )
    else:
        st.info("No positive association rules found " "for this basket.")
else:
    st.info("Add food items to your basket to get suggestions.")

st.divider()
st.header("🔎 Association Rules Explorer")

search_food = st.selectbox(
    "Select a food to explore its associations:",
    options=food_items,
    index=None,
    placeholder="Search for a food...",
)

if search_food:

    matching_rules = rules[
        rules["antecedents"].apply(lambda x: search_food in x.split(", "))
        | rules["consequents"].apply(lambda x: search_food in x.split(", "))
    ].copy()

    matching_rules = matching_rules[matching_rules["lift"] > 1].sort_values(
        "lift", ascending=False
    )

    st.write(
        f"Found **{len(matching_rules)}** positive "
        f"association rules involving **{search_food}**."
    )

    if not matching_rules.empty:

        st.dataframe(
            matching_rules[
                ["antecedents", "consequents", "support", "confidence", "lift"]
            ].round(4),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("No positive association rules found " "for this food.")

else:
    st.info("Select a food to explore its association rules.")
