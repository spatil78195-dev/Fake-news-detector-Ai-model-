"""
generate_dataset.py
-------------------
Generates a larger, more varied synthetic dataset for the Fake News Detector.
Creates 1500 fake + 1500 real articles (3000 total) for better model accuracy.

Run this BEFORE train_model.py if you don't have real Kaggle CSVs.
"""

import os
import csv
import random

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
FAKE_PATH   = os.path.join(DATASET_DIR, "Fake.csv")
TRUE_PATH   = os.path.join(DATASET_DIR, "True.csv")

# ---------------------------------------------------------------------------
# Expanded synthetic news stubs (more variety = better model)
# ---------------------------------------------------------------------------
FAKE_HEADLINES = [
    "SHOCKING: Scientists discover that 5G towers control human minds",
    "Government admits to spraying chemtrails to control population",
    "Bill Gates plans to microchip every human through COVID vaccine",
    "NASA fakes moon landing — leaked documents prove hoax",
    "Pope Francis secretly endorses presidential candidate",
    "Aliens have landed in Arizona and government is hiding the truth",
    "Clinton emails reveal plan for New World Order takeover",
    "Fluoride in water supply is mind-control chemical says whistleblower",
    "George Soros funding antifa to start civil war",
    "Deep state plotting to assassinate president claims insider",
    "Doctors confirm COVID-19 vaccine turns people magnetic",
    "Pentagon UFO footage shows alien crafts over Washington DC",
    "Secret tunnels under White House used for trafficking ring",
    "Earthquake machine HAARP caused Turkey disaster on purpose",
    "New study shows masks cause more deaths than COVID itself",
    "CIA admits using mainstream media to spread propaganda for decades",
    "Obama planned to declare martial law before 2016 election",
    "Rockefellers paid scientists to invent climate change hoax",
    "Q-Anon confirms deep state arrests imminent within 48 hours",
    "Epstein alive in witness protection says anonymous FBI source",
    "EXCLUSIVE: White House chef reveals president has dementia",
    "Breaking: Major city water supply contaminated with mind-altering drug",
    "Russia hacks US election server in real time, whistleblower says",
    "Big Pharma suppressing cancer cure for decades to make profits",
    "Hollywood elites caught performing satanic rituals in leaked video",
    "UN planning to ban meat, force citizens to eat insects by 2030",
    "Walmart tunnels are FEMA concentration camps for American citizens",
    "FDA approved poison in cereal brands given to children",
    "Prince Harry reveals royal family are shapeshifting reptilians",
    "Economy about to collapse, insiders moving gold to bunkers",
    "Doctors say sunscreen is poison pushed by Big Pharma",
    "Liberal elite planning to steal your guns using fake school shootings",
    "Hidden camera shows politicians admitting climate change is a hoax",
    "Chemotherapy cures nothing — industry invented it to steal your money",
    "Elon Musk's Neuralink is secretly harvesting human thoughts",
    "World leaders filmed at satanic ritual at Bohemian Grove",
    "Secret documents show GMO foods cause cancer in 10 years",
    "George Soros caught funding mass migration to destroy Western culture",
    "Mainstream media blackout on earthquake weapon test over California",
    "Covid-19 created in lab to usher in Great Reset admits Nobel scientist",
    "Elites using Hollywood movies to program the public for agenda",
    "Deep state agents infiltrated school boards across America",
    "Shadow government running secret operations beneath major airports",
    "New 5G antenna towers secretly contain surveillance cameras",
    "Bill Clinton admits moon landing staged to win space race",
    "FBI whistleblower reveals JFK assassination was inside job",
    "World Economic Forum plans to abolish private property by 2030",
    "Leaked Pfizer documents reveal vaccine contains aborted fetal cells",
    "Scientists confirm microplastics in vaccines used for population control",
    "Elite bankers using cryptocurrency to crash global economy on purpose",
]

REAL_HEADLINES = [
    "Federal Reserve raises interest rates by 25 basis points amid inflation concerns",
    "Scientists achieve breakthrough in Alzheimer's disease treatment in new study",
    "United Nations calls for ceasefire as conflict in region intensifies",
    "Apple reports record quarterly earnings driven by iPhone sales",
    "NASA successfully launches Artemis mission to lunar orbit",
    "WHO warns of rising antibiotic resistance as global health threat",
    "Congress passes bipartisan infrastructure bill worth 1.2 trillion dollars",
    "Climate summit reaches agreement on carbon emission reductions",
    "Stock markets decline as investors react to rising bond yields",
    "Pentagon releases annual defense budget request to Congress",
    "Supreme Court hears oral arguments in landmark free speech case",
    "European Central Bank signals potential interest rate cut next quarter",
    "CDC updates COVID-19 guidelines for healthcare workers",
    "Senate confirms new ambassador to key ally in diplomatic shift",
    "Tech companies face antitrust scrutiny from regulators in EU and US",
    "Global food prices rise as drought affects major agricultural regions",
    "International Monetary Fund revises global growth forecast downward",
    "President signs executive order on artificial intelligence regulation",
    "Oil prices fall after OPEC announces production increase",
    "New cancer screening program launches in several US states",
    "G20 leaders agree on framework for global minimum corporate tax",
    "SpaceX successfully lands booster for record fifteenth time",
    "Federal budget deficit widens as spending outpaces revenue growth",
    "Congressional hearing examines social media impact on teen mental health",
    "World Bank approves emergency loans to three developing nations",
    "Unemployment rate falls to lowest level in two years",
    "Researchers publish findings on sleep deprivation and cognitive decline",
    "NATO allies agree to increase defense spending commitments",
    "State department issues travel advisory for Southeast Asian region",
    "University study links ultra-processed foods to higher mortality risk",
    "Treasury Department proposes new tax reform package for small businesses",
    "Federal court rules against merger of two major airline carriers",
    "National Science Foundation awards grants for quantum computing research",
    "Defense Secretary meets with allies to discuss regional security threats",
    "Consumer confidence index rises for third consecutive month",
    "Health officials report significant decline in childhood obesity rates",
    "House passes bipartisan legislation on prescription drug pricing reform",
    "Agriculture Department releases annual crop yield forecast for winter",
    "Central bank governor testifies before Senate banking committee",
    "Environmental Protection Agency announces new clean air standards",
    "Transportation Secretary unveils plan to modernize air traffic control",
    "Census Bureau releases updated population projections for next decade",
    "Labor Department reports strong job gains across manufacturing sector",
    "Justice Department files antitrust suit against major technology firm",
    "Federal Reserve chair says inflation showing signs of gradual easing",
    "Science journal publishes breakthrough research on gene editing safety",
    "Presidential press secretary addresses reporters on foreign policy",
    "National Institutes of Health announces increased funding for cancer research",
    "Secretary of State holds bilateral talks with foreign minister in capital",
    "Commerce Department reports rise in trade deficit for second quarter",
]

FAKE_BODIES = [
    (
        "According to unnamed sources close to the situation, what mainstream media won't tell you is that {headline}. "
        "Our trusted insider, who cannot be named for their safety, confirmed this shocking revelation. "
        "The government has been working overtime to suppress this information. Share before it gets deleted! "
        "Independent researchers have long suspected this but were silenced by the establishment. "
        "The evidence is overwhelming yet the lamestream media refuses to report it. Wake up sheeple!"
    ),
    (
        "BREAKING — A leaked document obtained by this publication proves that {headline}. "
        "Officials deny any knowledge of the situation but sources say otherwise. "
        "This is the story they don't want you to see. Thousands of accounts spreading this news are being shadow-banned. "
        "Do your own research. The truth is out there but you have to look beyond the mainstream narrative. "
        "Forward this to everyone you know before censorship takes this down."
    ),
    (
        "Whistleblowers inside the agency have come forward to confirm that {headline}. "
        "The deep state operatives responsible for this crime are being protected at the highest levels of government. "
        "Patriotic citizens are urged to stand up and demand answers from their elected representatives. "
        "This is not a conspiracy theory — this is documented fact that they are hiding from the public. "
        "The truth will come out and when it does the globalists will have nowhere to run."
    ),
    (
        "An anonymous source with direct knowledge of the situation has confirmed that {headline}. "
        "The mainstream media cartel refuses to cover this bombshell story because they are complicit. "
        "Fact-checkers have already labeled this story as false — which is exactly what they do to hide truth. "
        "Real patriots already know the truth. The elite are running scared as this information goes viral. "
        "SHARE THIS NOW before Big Tech buries it! This is the information they are terrified of you knowing."
    ),
    (
        "URGENT: Multiple credible insiders have independently verified that {headline}. "
        "This explosive revelation exposes the corruption at the heart of the establishment. "
        "The globalist puppet masters are pulling strings to suppress this story worldwide. "
        "Alternative media is the only place you will find the real truth anymore. "
        "Our founding fathers warned us about exactly this kind of tyranny. The time to wake up is NOW."
    ),
]

REAL_BODIES = [
    (
        "In a statement released {day}, officials confirmed that {headline}. "
        "The decision follows months of negotiations and represents a significant policy shift. "
        "Experts said the move was expected given recent economic indicators and geopolitical developments. "
        'Spokesperson for the agency noted: "We are committed to transparency and will continue to update the public." '
        "The announcement was welcomed by industry leaders while some advocacy groups called for further review."
    ),
    (
        "{headline}, according to data released by the agency on {day}. "
        "The figures mark a notable change compared to the same period last year. "
        "Analysts attributed the shift to several macroeconomic factors including supply chain normalization. "
        "A senior official speaking on background said policymakers are closely monitoring the situation. "
        "Further details are expected to be released at a press briefing scheduled for later this week."
    ),
    (
        "Researchers at a leading institution published findings {day} showing that {headline}. "
        "The peer-reviewed study, conducted over 18 months, analyzed data from more than 10,000 participants. "
        "Lead author stated the results have important implications for public policy and future research. "
        "The methodology was independently verified by two external review boards before publication. "
        "Government agencies said they would consider the findings in upcoming regulatory reviews."
    ),
    (
        "The administration announced {day} that {headline}. "
        "Officials cited data showing consistent trends in the relevant sectors over the past quarter. "
        "Congressional leaders from both parties acknowledged the significance of the development. "
        'The press secretary stated: "This reflects our ongoing commitment to evidence-based policymaking." '
        "International observers noted the decision aligns with broader global economic trends."
    ),
    (
        "According to a report published {day}, {headline}. "
        "The findings were based on a comprehensive review of publicly available data and expert testimony. "
        "Committee members questioned agency officials for several hours before releasing the final report. "
        "Independent economists described the assessment as thorough and methodologically sound. "
        "The report will now be reviewed by the relevant oversight body before any formal policy action is taken."
    ),
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def generate_article(headline: str, bodies: list) -> str:
    template = random.choice(bodies)
    return template.format(headline=headline.lower(), day=random.choice(DAYS))


def write_csv(path: str, headlines: list, bodies: list, repeats: int = 1500):
    """Write a CSV with 'title', 'text', 'subject', 'date' columns."""
    subjects = ["politicsNews", "worldnews", "News", "left-news", "politics", "Government News"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "text", "subject", "date"])
        for _ in range(repeats):
            h = random.choice(headlines)
            b = generate_article(h, bodies)
            writer.writerow([
                h,
                b,
                random.choice(subjects),
                f"{random.randint(2016,2023)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            ])
    print(f"  Created {path}  ({repeats} rows)")


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)

    if os.path.exists(FAKE_PATH) and os.path.exists(TRUE_PATH):
        # Check if the existing files are the large version (>1000 rows)
        with open(FAKE_PATH, "r", encoding="utf-8") as f:
            row_count = sum(1 for _ in f) - 1  # subtract header

        if row_count >= 1000:
            print(f"[OK] Dataset files already exist ({row_count} rows each). Skipping generation.")
            return
        else:
            print(f"[INFO] Existing dataset too small ({row_count} rows). Regenerating with {1500} rows...")

    print("Generating synthetic dataset (use real Kaggle CSVs for production)...")
    random.seed(42)
    write_csv(FAKE_PATH, FAKE_HEADLINES, FAKE_BODIES, repeats=1500)
    write_csv(TRUE_PATH, REAL_HEADLINES, REAL_BODIES, repeats=1500)
    print("[OK] Synthetic dataset generated successfully (3000 total samples).")
    print("     TIP: For best accuracy, replace these with the ISOT Fake News dataset from Kaggle.")


if __name__ == "__main__":
    main()
