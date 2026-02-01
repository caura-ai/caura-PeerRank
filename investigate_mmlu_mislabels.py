"""
MMLU Mislabel Investigation

Investigates potential mislabeled questions in MMLU dataset, focusing on the medical subset.
Looks for:
- Physics questions in non-physics subjects
- Math questions in non-math subjects
- Questions that obviously don't match their labeled subject

Usage:
    python investigate_mmlu_mislabels.py
"""

import re
from collections import defaultdict
from datasets import load_dataset

# Medical subset subjects to investigate
MEDICAL_SUBJECTS = [
    "clinical_knowledge",
    "medical_genetics",
    "anatomy",
    "professional_medicine",
    "college_medicine",
    "college_biology",
    "virology",
    "nutrition",
    "human_aging",
]

# Pure physics problems - these require physics calculations, not medical knowledge
# More specific patterns to avoid false positives
PHYSICS_PROBLEM_PATTERNS = [
    # Inclined plane problems
    (r'(plane|surface).*angle.*incline', "inclined plane physics problem"),
    (r'incline.*angle.*coefficient', "inclined plane physics problem"),
    (r'coefficient.*friction.*angle', "friction on incline problem"),
    (r'ramp.*angle.*friction', "inclined plane problem"),

    # Kinematics problems (car/object motion)
    (r'(car|object|block).*accelerat.*reach.*velocity', "kinematics calculation"),
    (r'(car|object).*velocity.*distance.*rate', "kinematics problem"),
    (r'distance.*track.*velocity.*accelerat', "kinematics calculation"),
    (r'must.*accelerate.*reach.*velocity', "kinematics problem"),

    # Sound/wave physics (not medical ultrasound context)
    (r'sound.*exits.*medium.*enters.*denser', "wave physics problem"),
    (r'sound.*medium.*velocity.*wavelength', "wave physics problem"),

    # Fluid dynamics (not medical blood flow)
    (r'fire\s*hose.*nozzle.*velocity', "fluid dynamics problem"),
    (r'hose.*nozzle.*area.*velocity', "Bernoulli/continuity problem"),
    (r'hydrant.*velocity.*area', "fluid dynamics problem"),

    # Generic physics setup
    (r'race\s*car.*jump.*velocity', "projectile motion problem"),
    (r'object\s*rests.*plane.*angle.*acceleration', "inclined plane problem"),
]

# Questions that look medical but are actually pure physics calculations
def is_pure_physics_problem(question_text, choices):
    """Check if this is a pure physics problem mislabeled as medical."""
    full_text = (question_text + " " + " ".join(choices)).lower()

    for pattern, description in PHYSICS_PROBLEM_PATTERNS:
        if re.search(pattern, full_text, re.IGNORECASE):
            # Additional check: physics answers (m/s^2, m/s, etc.) suggest physics problem
            physics_units = re.search(r'm/s(\^2)?|km/h|N\b|kg\b', " ".join(choices))
            if physics_units:
                return (True, description, pattern)
            # Also check for pure letter/number physics answers
            physics_format = all(re.match(r'^[0-9\.\s]+m/s', c.strip()) or
                                re.match(r'^[0-9]+\s*(m/s|N|kg|J|W)', c.strip()) or
                                re.match(r'a\s*=\s*g', c.strip().lower())
                                for c in choices)
            if physics_format:
                return (True, description, pattern)
            # For certain patterns, context is enough
            if 'incline' in pattern or 'kinematics' in description or 'fire hose' in description:
                return (True, description, pattern)

    return (False, None, None)


def analyze_question_context(question_text, labeled_subject, choices):
    """Deeper analysis of question context to find mislabels."""
    full_text = (question_text + " " + " ".join(choices)).lower()
    findings = []

    # Check for pure physics problems in medical subjects
    is_physics, description, pattern = is_pure_physics_problem(question_text, choices)
    if is_physics:
        findings.append({
            "detected_subject": "PHYSICS",
            "description": description,
            "pattern": pattern,
            "confidence": "HIGH"
        })

    # Check for MCAT-style physics in college_medicine
    # MCAT includes physics, so these may be intentionally included but labeled wrong
    if labeled_subject == "college_medicine":
        # Check if it's clearly a physics problem context
        mcat_physics_indicators = [
            r'(\d+)\s*(m/s|km/h).*(\d+)\s*m\b',  # velocity and distance
            r'angle.*\d+\s*degrees?.*friction',   # angle + friction
            r'acceleration\s*due\s*to\s*gravity',
            r'coefficient\s*of.*friction',
        ]
        for pattern in mcat_physics_indicators:
            if re.search(pattern, full_text):
                # Verify it's not in a medical context
                medical_context = re.search(r'patient|blood|heart|lung|disease|symptom|treatment', full_text)
                if not medical_context:
                    if not any(f["detected_subject"] == "PHYSICS" for f in findings):
                        findings.append({
                            "detected_subject": "PHYSICS (MCAT-style)",
                            "description": "Physics calculation without medical context",
                            "pattern": pattern,
                            "confidence": "HIGH"
                        })

    return findings


def search_for_physics_patterns(dataset, subjects):
    """Search all questions in subjects for physics-like patterns."""
    physics_indicators = [
        # Kinematics/mechanics
        (r'\bm/s\^?2\b', 'has m/s^2 unit'),
        (r'\bkm/h\b.*\bm\b', 'has km/h and meters'),
        (r'coefficient\s*(of)?\s*(static|kinetic)?\s*friction', 'friction coefficient'),
        (r'angle\s*(of)?\s*inclin', 'angle of incline'),
        (r'\binclined?\s*plane', 'inclined plane'),
        (r'accelerat.*reach.*velocit', 'acceleration-velocity'),
        (r'velocit.*accelerat', 'velocity-acceleration'),
        (r'projectile\s*motion', 'projectile motion'),

        # Fluid dynamics (non-medical)
        (r'(fire\s*)?hose.*nozzle', 'hose/nozzle fluid dynamics'),
        (r'Bernoulli', 'Bernoulli equation'),
        (r'continuity\s*equation', 'continuity equation'),

        # Electromagnetism
        (r'electric\s*field.*charge', 'electric field'),
        (r'magnetic\s*field.*current', 'magnetic field'),
        (r'Ohm.*law', 'Ohm law'),

        # Thermodynamics (non-biological)
        (r'ideal\s*gas\s*law', 'ideal gas law'),
        (r'adiabatic\s*process', 'adiabatic process'),
        (r'Carnot\s*(cycle|engine)', 'Carnot cycle'),

        # Waves/optics
        (r'sound.*medium.*velocity', 'sound wave physics'),
        (r'refraction.*index', 'optics refraction'),
    ]

    results = []
    for row in dataset:
        subject = row.get("subject", "unknown")
        if subject not in subjects:
            continue

        full_text = row["question"].lower() + " " + " ".join(row["choices"]).lower()

        for pattern, desc in physics_indicators:
            if re.search(pattern, full_text, re.IGNORECASE):
                # Check if medical context
                medical_context = re.search(
                    r'patient|blood\s*(flow|pressure)|heart|cardiac|lung|pulmonary|disease|symptom|treatment|diagnosis|artery|vein|capillary|tissue|cell|organ',
                    full_text, re.IGNORECASE
                )
                if not medical_context:
                    results.append({
                        "subject": subject,
                        "question": row["question"],
                        "choices": row["choices"],
                        "answer": row["answer"],
                        "pattern": desc,
                    })
                    break
    return results


def search_all_mcat_physics(dataset):
    """
    The MCAT has physics sections. college_medicine likely contains MCAT prep questions.
    Search for all physics-style questions that have no medical relevance.
    """
    # Typical physics question topics that appear on MCAT but aren't medicine
    physics_topics = {
        'kinematics': [
            r'velocity.*m/s',
            r'm/s\^?2',
            r'acceleration.*rate',
            r'distance.*time.*speed',
            r'projectile',
        ],
        'mechanics': [
            r'incline.*angle',
            r'angle.*incline',
            r'coefficient.*friction',
            r'friction.*coefficient',
            r'force.*mass.*acceleration',
            r'Newton.*law',
            r'momentum.*collision',
            r'torque.*rotation',
            r'work.*energy.*joule',
            r'spring.*constant',
            r'pendulum.*period',
        ],
        'fluids': [
            r'(fire\s*)?hose.*velocity',
            r'nozzle.*velocity',
            r'Bernoulli',
            r'continuity.*equation',
            r'buoyancy',
            r'fluid.*pressure.*depth',
            r'Archimedes',
        ],
        'waves_sound': [
            r'sound.*medium.*velocity',
            r'sound.*exits.*medium',
            r'wavelength.*frequency',
            r'Doppler.*effect',
            r'standing.*wave',
            r'resonance.*frequency',
        ],
        'optics': [
            r'lens.*focal.*length',
            r'refraction.*index',
            r'mirror.*image',
            r'diffraction',
        ],
        'electromagnetism': [
            r'electric.*field.*charge',
            r'capacitor.*capacitance',
            r'resistor.*circuit',
            r'Ohm.*law',
            r'magnetic.*field.*wire',
        ],
        'thermodynamics': [
            r'ideal.*gas.*law',
            r'heat.*capacity',
            r'adiabatic',
            r'isothermal',
            r'Carnot',
            r'entropy.*process',
        ],
    }

    results = []
    for row in dataset:
        subject = row.get("subject", "unknown")
        if subject != "college_medicine":
            continue

        full_text = row["question"].lower() + " " + " ".join(row["choices"]).lower()

        # Check if medical context exists
        medical_context = re.search(
            r'patient|blood\s*(flow|pressure|cell)|heart|cardiac|lung|pulmonary|disease|symptom|treatment|diagnosis|artery|vein|capillary|tissue|cell\s*(membrane|biology)|organ|muscle|bone|brain|nerve|neuron|hormone|enzyme|protein\s*(synthesis|function)|DNA|RNA|gene|metabolism|respiration\s*(rate)?$|immune|antibody|vaccine|infection|bacteria|virus(?!.*computer)|surgery|medication|drug\s*(effect|interaction)|clinical|medical|physician|doctor|hospital',
            full_text, re.IGNORECASE
        )

        for topic, patterns in physics_topics.items():
            for pattern in patterns:
                if re.search(pattern, full_text, re.IGNORECASE):
                    if not medical_context:
                        results.append({
                            "subject": subject,
                            "topic": topic,
                            "pattern": pattern,
                            "question": row["question"],
                            "choices": row["choices"],
                            "answer": row["answer"],
                        })
                    break
            else:
                continue
            break

    return results


def manual_review_college_medicine(dataset):
    """
    Manually review all college_medicine questions to find obvious mislabels.
    Returns questions that have zero medical terminology.
    """
    medical_terms = [
        'patient', 'blood', 'heart', 'cardiac', 'lung', 'pulmonary', 'disease',
        'symptom', 'treatment', 'diagnosis', 'artery', 'vein', 'tissue', 'cell',
        'organ', 'muscle', 'bone', 'brain', 'nerve', 'neuron', 'hormone', 'enzyme',
        'protein', 'DNA', 'RNA', 'gene', 'metabolism', 'immune', 'antibody',
        'vaccine', 'infection', 'bacteria', 'virus', 'surgery', 'medication',
        'drug', 'clinical', 'medical', 'physician', 'doctor', 'hospital',
        'anatomy', 'physiology', 'pathology', 'pharmacology', 'therapy',
        'syndrome', 'disorder', 'condition', 'health', 'body', 'human',
        'biological', 'cellular', 'molecular', 'genetic', 'biochem',
        'MCAT', 'premed', 'medical school', 'electrochemical', 'redox',
        'amino acid', 'peptide', 'membrane', 'receptor', 'ligand',
        'psychology', 'cogniti', 'behavior', 'emotion', 'memory', 'learning',
        'stimulus', 'response', 'perception', 'attention', 'conscious'
    ]

    suspicious = []
    for row in dataset:
        subject = row.get("subject", "unknown")
        if subject != "college_medicine":
            continue

        full_text = row["question"].lower() + " " + " ".join(row["choices"]).lower()

        # Check if ANY medical/biology term exists
        has_medical = False
        for term in medical_terms:
            if term.lower() in full_text:
                has_medical = True
                break

        if not has_medical:
            suspicious.append({
                "question": row["question"],
                "choices": row["choices"],
                "answer": row["answer"],
            })

    return suspicious


def main():
    print("Loading MMLU dataset from HuggingFace...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    print(f"Total questions: {len(dataset)}")

    # Filter to medical subjects
    print(f"\nFiltering to medical subjects: {MEDICAL_SUBJECTS}")

    by_subject = defaultdict(list)
    for row in dataset:
        subject = row.get("subject", "unknown")
        if subject in MEDICAL_SUBJECTS:
            by_subject[subject].append({
                "question": row["question"],
                "choices": row["choices"],
                "answer": row["answer"],
                "subject": subject,
            })

    total_medical = sum(len(v) for v in by_subject.values())
    print(f"Total medical questions: {total_medical}")
    print()

    # Show counts per subject
    print("Questions per subject:")
    for subject in MEDICAL_SUBJECTS:
        count = len(by_subject.get(subject, []))
        print(f"  {subject}: {count}")
    print()

    # Method 0: Manual review - find questions with NO medical terminology at all
    print("=" * 80)
    print("MANUAL REVIEW: Questions with no medical terminology")
    print("=" * 80)

    no_medical = manual_review_college_medicine(dataset)
    print(f"Found {len(no_medical)} questions with no medical/biology terminology")

    if no_medical:
        print("\nQuestions with zero medical context:")
        for i, item in enumerate(no_medical, 1):
            # Handle encoding issues
            q = item['question'].encode('ascii', 'replace').decode('ascii')
            choices = [c.encode('ascii', 'replace').decode('ascii') for c in item['choices']]
            print(f"\n--- {i}. ---")
            print(f"Q: {q}")
            print(f"Choices: {choices}")
            letters = "ABCD"
            print(f"Correct: {letters[item['answer']]}")

    # Method 1: Comprehensive MCAT physics search in college_medicine
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SEARCH: Physics questions in college_medicine")
    print("=" * 80)

    all_physics = search_all_mcat_physics(dataset)
    print(f"Found {len(all_physics)} physics questions without medical context in college_medicine")

    # Group by topic
    by_topic = defaultdict(list)
    for item in all_physics:
        by_topic[item['topic']].append(item)

    for topic in sorted(by_topic.keys()):
        items = by_topic[topic]
        print(f"\n  {topic.upper()}: {len(items)} questions")
        for item in items:
            print(f"    - {item['question'][:80]}...")

    # Method 2: Broader pattern search
    print("\n" + "=" * 80)
    print("SEARCHING FOR PHYSICS PATTERNS (all medical subjects)...")
    print("=" * 80)

    physics_hits = search_for_physics_patterns(dataset, MEDICAL_SUBJECTS)
    print(f"Found {len(physics_hits)} potential physics questions in all medical subjects")

    if physics_hits:
        print("\nQuestions with physics indicators (no medical context):")
        for i, hit in enumerate(physics_hits, 1):
            print(f"\n--- {i}. {hit['subject']} ---")
            print(f"Pattern: {hit['pattern']}")
            print(f"Q: {hit['question'][:200]}...")
            print(f"Choices: {hit['choices']}")

    print("\n" + "=" * 80)
    print("CONFIRMED MISLABELS (high confidence):")
    print("=" * 80)

    # Method 2: Original detailed analysis
    mislabeled = []

    for subject, questions in by_subject.items():
        for q in questions:
            findings = analyze_question_context(q["question"], subject, q["choices"])
            if findings:
                mislabeled.append({
                    "subject": subject,
                    "question": q["question"],
                    "choices": q["choices"],
                    "correct_answer": q["answer"],
                    "findings": findings,
                })

    # Report findings
    print("=" * 80)
    print(f"CONFIRMED MISLABELS FOUND: {len(mislabeled)}")
    print("=" * 80)

    # Group by detected mismatch type
    by_type = defaultdict(list)
    for item in mislabeled:
        for finding in item["findings"]:
            by_type[finding["detected_subject"]].append(item)

    for mtype in sorted(by_type.keys()):
        items = by_type[mtype]
        print(f"\n{'=' * 60}")
        print(f"{mtype} questions mislabeled as medical: {len(items)}")
        print("=" * 60)

        # Remove duplicates
        seen_questions = set()
        for item in items:
            if item["question"] in seen_questions:
                continue
            seen_questions.add(item["question"])

            print(f"\n{'-' * 60}")
            print(f"LABELED AS: {item['subject']}")
            print(f"SHOULD BE:  {mtype}")
            print(f"{'-' * 60}")
            print(f"Question: {item['question']}")
            print(f"\nChoices:")
            letters = "ABCD"
            for j, choice in enumerate(item['choices']):
                marker = " (*)" if j == item['correct_answer'] else ""
                print(f"  {letters[j]}. {choice}{marker}")
            print(f"\nWhy mislabeled: {item['findings'][0]['description']}")

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\nTotal medical questions analyzed: {total_medical}")
    print(f"Total college_medicine questions: {len(by_subject.get('college_medicine', []))}")
    print(f"\nQuestions with NO medical terminology: {len(no_medical)}")
    print(f"  - These are likely MCAT prep questions mislabeled as medicine")
    print(f"  - Should be categorized by their actual subject area")
    print(f"\nHigh-confidence physics mislabels: {len(mislabeled)}")

    # Categorize the no-medical questions
    print("\n" + "=" * 80)
    print("CATEGORIZATION OF MISLABELED QUESTIONS")
    print("=" * 80)

    categories = {
        'PHYSICS': [],
        'CHEMISTRY': [],
        'PSYCHOLOGY': [],
        'SOCIOLOGY': [],
        'BIOCHEMISTRY': [],
        'EXERCISE_PHYSIOLOGY': [],
        'GENETICS': [],
        'TRIVIA': [],
        'OTHER': []
    }

    # Manual categorization keywords
    for item in no_medical:
        q = item['question'].lower()
        choices_text = " ".join(item['choices']).lower()
        full = q + " " + choices_text

        if any(x in full for x in ['m/s', 'velocity', 'acceleration', 'incline', 'friction', 'sound medium', 'doppler', 'nozzle', 'hose']):
            categories['PHYSICS'].append(item)
        elif any(x in full for x in ['acid', 'base', 'pkb', 'ka', 'hclo4', 'stoichiometry', 'moles', 'chromatography', 'bromobutane', 'electrons', 'quantum', 'mg(oh)']):
            categories['CHEMISTRY'].append(item)
        elif any(x in full for x in ['psychologist', 'freud', 'attachment', 'prejudice', 'bystander', 'dependent variable', 'cogniti']):
            categories['PSYCHOLOGY'].append(item)
        elif any(x in full for x in ['folkways', 'mores', 'taboo', 'society', 'ethnocentrism', 'task force']):
            categories['SOCIOLOGY'].append(item)
        elif any(x in full for x in ['glycolysis', 'krebs', 'atp', 'fadh', 'nadh', 'pyruvate', 'acetyl', 'oxidation']):
            categories['BIOCHEMISTRY'].append(item)
        elif any(x in full for x in ['exercise', 'marathon', 'phosphocreatine', 'aerobic', 'oxygen consumption', 'kj', 'stamina']):
            categories['EXERCISE_PHYSIOLOGY'].append(item)
        elif any(x in full for x in ['chromosome', 'sex of a child', 'y chromosome']):
            categories['GENETICS'].append(item)
        elif any(x in full for x in ['world record', 'mile race', '1886']):
            categories['TRIVIA'].append(item)
        else:
            categories['OTHER'].append(item)

    for cat, items in categories.items():
        if items:
            print(f"\n{cat}: {len(items)} questions")
            for item in items[:3]:  # Show first 3 examples
                q = item['question'][:100].encode('ascii', 'replace').decode('ascii')
                print(f"  - {q}...")
            if len(items) > 3:
                print(f"  ... and {len(items)-3} more")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(f"""
The MMLU dataset's "college_medicine" subject contains {len(no_medical)} questions
({100 * len(no_medical) / len(by_subject.get('college_medicine', [1])):.1f}% of college_medicine)
that appear to be mislabeled.

These questions cover:
- Physics ({len(categories['PHYSICS'])} questions): kinematics, mechanics, waves, fluids
- Chemistry ({len(categories['CHEMISTRY'])} questions): acids/bases, organic chemistry
- Psychology ({len(categories['PSYCHOLOGY'])} questions): developmental, social psychology
- Sociology ({len(categories['SOCIOLOGY'])} questions): norms, deviance
- Biochemistry ({len(categories['BIOCHEMISTRY'])} questions): metabolic pathways
- Exercise Physiology ({len(categories['EXERCISE_PHYSIOLOGY'])} questions): energy systems
- Genetics ({len(categories['GENETICS'])} questions): inheritance
- Trivia ({len(categories['TRIVIA'])} questions): sports history
- Other ({len(categories['OTHER'])} questions): uncategorized

The likely cause: These are MCAT preparation questions that were incorrectly
categorized as "college_medicine" instead of their actual subject areas.
The MCAT exam includes sections on physics, chemistry, psychology, and sociology,
which explains why these non-medical questions ended up in this category.
""")


if __name__ == "__main__":
    main()
