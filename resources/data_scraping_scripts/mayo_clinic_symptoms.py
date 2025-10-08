import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define blacklisted words to remove crisis hotlines and unrelated sections
blacklisted_keywords = ["Crisis", "Emergency", "Hotline", "988", "Veterans Crisis Line", "Intensity of symptoms", "Products & Services"]

def scrape_mayo_symptoms(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to retrieve {url}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract disorder title
    title = soup.find("h1").text.strip() if soup.find("h1") else "Unknown"

    symptom_sections = []
    
    # Find the main "Symptoms" section (h2)
    for section in soup.find_all("h2"):
        if "symptom" in section.text.lower():
            symptom_category = section.text.strip()  # Main Symptoms category

            # Look for subcategories (h3 under h2)
            subcategory = None
            for sub_section in section.find_all_next(["h3", "ul"], limit=10):  # Find nearby h3 (subcategories) or ul (symptoms list)
                if sub_section.name == "h3":  # If it's a subcategory, update variable
                    subcategory = sub_section.text.strip()

                    # Skip blacklisted sections
                    if any(keyword.lower() in subcategory.lower() for keyword in blacklisted_keywords):
                        subcategory = None  # Reset subcategory to avoid storing irrelevant resources
                        continue

                elif sub_section.name == "ul" and subcategory:  # Symptoms list under a subcategory
                    symptoms = [li.text.strip() for li in sub_section.find_all("li")]
                    symptom_sections.append({"category": symptom_category, "subcategory": subcategory, "symptoms": symptoms})

    return {"title": title, "symptom_sections": symptom_sections}

# Test with PTSD page
url = "https://www.mayoclinic.org/diseases-conditions/post-traumatic-stress-disorder/symptoms-causes/syc-20355967"
data = scrape_mayo_symptoms(url)

import pprint
pprint.pprint(data)

# List of Mayo Clinic URLs
urls = [
    "https://www.mayoclinic.org/diseases-conditions/adult-adhd/symptoms-causes/syc-20350878",
    "https://www.mayoclinic.org/diseases-conditions/mental-illness/symptoms-causes/syc-20374968",
    "https://www.mayoclinic.org/diseases-conditions/depression/symptoms-causes/syc-20356007",
    "https://www.mayoclinic.org/diseases-conditions/anxiety/symptoms-causes/syc-20350961",
    "https://www.mayoclinic.org/diseases-conditions/schizophrenia/symptoms-causes/syc-20354443",
    "https://www.mayoclinic.org/diseases-conditions/bipolar-disorder/symptoms-causes/syc-20355955",
    "https://www.mayoclinic.org/diseases-conditions/mood-disorders/symptoms-causes/syc-20365057",
    "https://www.mayoclinic.org/diseases-conditions/post-traumatic-stress-disorder/symptoms-causes/syc-20355967",
    "https://www.mayoclinic.org/diseases-conditions/obsessive-compulsive-disorder/symptoms-causes/syc-20354432",
    "https://www.mayoclinic.org/diseases-conditions/eating-disorders/symptoms-causes/syc-20353603",
    "https://www.mayoclinic.org/diseases-conditions/personality-disorders/symptoms-causes/syc-20354463",
    "https://www.mayoclinic.org/diseases-conditions/autism-spectrum-disorder/symptoms-causes/syc-20352928",
    "https://www.mayoclinic.org/diseases-conditions/dissociative-disorders/symptoms-causes/syc-20355215",
    "https://www.mayoclinic.org/diseases-conditions/sleep-disorders/symptoms-causes/syc-20354018",
    "https://www.mayoclinic.org/diseases-conditions/drug-addiction/symptoms-causes/syc-20365112",
    "https://www.mayoclinic.org/diseases-conditions/self-injury/symptoms-causes/syc-20350950",
    "https://www.mayoclinic.org/diseases-conditions/suicide/symptoms-causes/syc-20378048",
    "https://www.mayoclinic.org/diseases-conditions/oppositional-defiant-disorder/symptoms-causes/syc-20375831",
    "https://www.mayoclinic.org/diseases-conditions/mild-cognitive-impairment/symptoms-causes/syc-20354578"
]

# Scrape resources from all pages
all_symptoms = []
for url in urls:
    result = scrape_mayo_symptoms(url)
    if result:
        all_symptoms.append(result)

# Convert to DataFrame and save to CSV
df = pd.DataFrame(all_symptoms)
df.to_csv("mayo_symptoms.csv", index=False)
