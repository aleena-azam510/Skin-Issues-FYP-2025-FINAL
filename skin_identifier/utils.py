import random

def diagnose_skin_type(data):
    """
    Diagnoses skin type and provides tailored advice based on user input.
    """
    # --- Helper Functions ---
    def get_random_product(category, skin_type):
        products = all_products.get(category, {}).get(skin_type, [])
        return random.choice(products) if products else None

    def get_multiple_products(category, skin_type, count):
        products = all_products.get(category, {}).get(skin_type, [])
        return random.sample(products, min(count, len(products))) if products else []

    def get_multiple_diy(skin_type, count):
        remedies = all_diy.get(skin_type, [])
        return random.sample(remedies, min(count, len(remedies))) if remedies else []

    def calculate_scores(data):
        scores = {"Dry": 0, "Oily": 0, "Combination": 0, "Normal": 0}
        
        # Oiliness scoring
        oiliness = data.get('oiliness', 'normal')
        if oiliness == 'oily':
            scores['Oily'] += 6
        elif oiliness == 'combination':
            scores['Combination'] += 4
        elif oiliness == 'normal':
            scores['Normal'] += 2
            
        # Dryness scoring
        dryness = data.get('dryness', 'rarely')
        if dryness == 'frequent':
            scores['Dry'] += 5
        elif dryness == 'sometimes':
            scores['Dry'] += 2
            
        # Pore size scoring
        pores = data.get('pores', 'small')
        if pores == 'large':
            scores['Oily'] += 3
        elif pores == 'mixed':
            scores['Combination'] += 2
            
        # Acne scoring
        skin_concerns = data.get('skin_concerns', [])
        if 'Acne or breakouts' in skin_concerns:
            scores['Oily'] += 4
            
        return scores

    # --- Product Database (with simplified descriptions) ---
    all_products = {
        "cleanser": {
            "Oily": [
                {"item": "Foaming Cleanser", "desc": "A foaming wash that deep cleans pores and helps with breakouts."},
                {"item": "Charcoal Cleanser", "desc": "This cleanser uses charcoal to pull out dirt and oil."},
                {"item": "Niacinamide Gel Cleanser", "desc": "Reduces oil and makes pores look smaller without drying out your skin."},
                {"item": "Sulfur Acne Wash", "desc": "A medicated wash that fights acne and controls extra oil."}
            ],
            "Dry": [
                {"item": "Creamy Hydrating Cleanser", "desc": "A gentle, creamy wash that removes dirt and adds moisture back to your skin."},
                {"item": "Oil-based Cleansing Balm", "desc": "Melts away makeup and dirt without stripping your skin's natural oils."},
                {"item": "Hyaluronic Acid Face Wash", "desc": "A moisturizing wash that helps your skin stay hydrated after you wash it."},
                {"item": "Oat Milk Cleansing Milk", "desc": "A soothing, milky formula for dry, irritated skin."}
            ],
            "Combination": [
                {"item": "Gentle Foaming Cleanser", "desc": "Cleanses without over-drying, perfect for both oily and dry spots."},
                {"item": "Micellar Water", "desc": "A simple, no-rinse option that cleans and tones without irritation."},
                {"item": "Tea Tree Oil Cleanser", "desc": "Helps with breakouts and oiliness on your T-zone while being gentle on drier areas."},
                {"item": "Balancing Glycolic Cleanser", "desc": "Gently exfoliates to make your skin smoother and clearer."}
            ],
            "Normal": [
                {"item": "Balanced pH Gentle Cleanser", "desc": "Maintains your skin's natural balance while cleaning well."},
                {"item": "Hydrating Milk Cleanser", "desc": "A soft, non-foaming wash that leaves your skin feeling fresh."},
                {"item": "Vitamin-rich Gel Cleanser", "desc": "Cleanses gently and gives your skin essential vitamins."},
                {"item": "Botanical Cream Cleanser", "desc": "Uses plant extracts to cleanse and protect your skin's barrier."}
            ],
            "Sensitive": [
                {"item": "Soothing Gentle Cleanser", "desc": "Made for sensitive skin to calm redness and irritation."},
                {"item": "Calamine & Zinc Cleansing Wash", "desc": "Reduces skin inflammation with a gentle, non-irritating wash."}
            ]
        },
        "toner": {
            "Oily": [
                {"item": "Salicylic Acid Toner", "desc": "Gets deep into pores to prevent breakouts and control oil."},
                {"item": "Witch Hazel Toner", "desc": "Helps make pores look smaller and reduces inflammation."},
                {"item": "Green Tea Toner", "desc": "Reduces oil and minimizes the look of pores."}
            ],
            "Dry": [
                {"item": "Hydrating Rosewater Toner", "desc": "Soothes and hydrates your skin, getting it ready for moisturizer."},
                {"item": "Hyaluronic Acid Toner", "desc": "Gives a quick boost of moisture to fight tight, dry skin."},
                {"item": "Ceramide-rich Facial Mist", "desc": "A refreshing spray that instantly helps strengthen your skin."}
            ],
            "Combination": [
                {"item": "Balancing Glycolic Acid Toner", "desc": "Gently exfoliates both oily and dry patches."},
                {"item": "Niacinamide Toner", "desc": "Reduces oil on your T-zone while calming dry areas."},
                {"item": "BHA and PHA Toner", "desc": "A gentle toner that helps clear out pores."}
            ],
            "Normal": [
                {"item": "Antioxidant-rich Green Tea Toner", "desc": "Protects and refreshes your skin."},
                {"item": "Vitamin C Toner", "desc": "Brightens your skin tone and gives protection from damage."},
                {"item": "Probiotic Balancing Toner", "desc": "Helps support a healthy balance of good bacteria on your skin."}
            ],
            "Sensitive": [
                {"item": "Alcohol-Free Calming Toner", "desc": "A gentle formula to reduce redness and avoid irritation."},
                {"item": "Cica Toner", "desc": "Calms and soothes sensitive skin."}
            ]
        },
        "moisturizer": {
            "Oily": [
                {"item": "Oil-Free Gel Moisturizer", "desc": "Hydrates your skin without feeling greasy."},
                {"item": "Mattifying Moisturizer with SPF", "desc": "Controls shine and protects you from the sun."},
                {"item": "Niacinamide Moisturizer", "desc": "Regulates oil and provides hydration."}
            ],
            "Dry": [
                {"item": "Rich Cream with Ceramides", "desc": "A thick cream that deeply hydrates and protects your skin's barrier."},
                {"item": "Nourishing Facial Oil", "desc": "Helps your skin lock in moisture with good fatty acids."},
                {"item": "Shea Butter Night Cream", "desc": "Provides intense relief for dry, tight skin while you sleep."}
            ],
            "Combination": [
                {"item": "Balancing Lightweight Lotion", "desc": "Hydrates dry areas without clogging pores."},
                {"item": "Water-based Gel Cream", "desc": "A light and refreshing cream for combination skin."},
                {"item": "Probiotic Balancing Moisturizer", "desc": "Helps balance both oily and dry areas."}
            ],
            "Normal": [
                {"item": "Everyday Hydrating Lotion", "desc": "Keeps your skin healthy and soft."},
                {"item": "Antioxidant-rich Facial Cream", "desc": "Protects your skin from daily stress."},
                {"item": "Squalane Moisturizer", "desc": "Lightweight but very effective moisture."}
            ],
            "Sensitive": [
                {"item": "Fragrance-Free Barrier Cream", "desc": "A calming cream that reduces redness and strengthens your skin's natural barrier."},
                {"item": "Ceramide Soothing Lotion", "desc": "Reduces sensitivity and makes your skin feel comfortable."}
            ]
        },
        "treatment": {
            "Acne": [
                {"item": "Benzoyl Peroxide Gel", "desc": "Kills acne-causing bacteria without making your skin too dry."},
                {"item": "Salicylic Acid Liquid", "desc": "Exfoliates inside pores to prevent new breakouts."},
                {"item": "Niacinamide + Zinc Serum", "desc": "Reduces inflammation and controls oil."},
                {"item": "Sulfur Spot Treatment", "desc": "Dries out active pimples quickly."}
            ],
            "Aging": [
                {"item": "Retinol Serum", "desc": "Boosts collagen to help reduce the look of fine lines."},
                {"item": "Vitamin C Serum", "desc": "Brightens your skin and protects it from damage."},
                {"item": "Peptide Complex Serum", "desc": "Supports your skin's elasticity and firmness."}
            ],
            "Sensitive": [
                {"item": "Cica Serum", "desc": "Calms and soothes irritated skin."},
                {"item": "Hyaluronic Acid Serum", "desc": "Provides deep, gentle hydration."}
            ],
            "General": [
                {"item": "Hyaluronic Acid Hydration Serum", "desc": "Adds moisture and plumps up all skin types."},
                {"item": "Alpha Arbutin Brightening Serum", "desc": "Gently evens out your skin tone."}
            ]
        },
        "sunscreen": {
            "all": [
                {"item": "Oil-Free SPF 50", "desc": "Protects your skin from the sun without making it look shiny."},
                {"item": "Mineral SPF 30", "desc": "A gentle formula that won't irritate sensitive skin."},
                {"item": "Tinted Moisturizer SPF 40", "desc": "Gives you light coverage with sun protection."}
            ]
        },
        "extra": {
            "Oily": [
                {"item": "Clay Mask", "desc": "Use once a week to absorb extra oil."},
                {"item": "Oil-blotting Sheets", "desc": "Perfect for quick shine control on the go."},
                {"item": "Pore-clearing Exfoliant", "desc": "Helps unclog pores and smooth out skin."}
            ],
            "Dry": [
                {"item": "Hydrating Sheet Mask", "desc": "A face mask for an intense boost of moisture."},
                {"item": "Sleeping Mask", "desc": "Provides nourishment for your skin overnight."}
            ],
            "Combination": [
                {"item": "Balancing Sheet Mask", "desc": "A mask that helps both oily and dry areas."},
                {"item": "Gentle Exfoliating Scrub", "desc": "Removes dead skin cells to improve texture."}
            ],
            "Normal": [
                {"item": "Brightening Mask", "desc": "Helps enhance your skin's natural glow."},
                {"item": "Hydrating Eye Cream", "desc": "Protects the delicate skin around your eyes."}
            ]
        }
    }

    all_diy = {
        "Oily": [
            {"item": "Clay & Tea Tree Mask", "desc": "Mix bentonite clay with a drop of tea tree oil and water. Apply for 10-15 minutes."},
            {"item": "Apple Cider Vinegar Toner", "desc": "Dilute 1 part vinegar to 3 parts water. Use a cotton pad to apply."},
            {"item": "Honey & Cinnamon Spot Treatment", "desc": "Mix a small amount of raw honey with cinnamon and dab on blemishes."}
        ],
        "Dry": [
            {"item": "Avocado & Honey Mask", "desc": "Mash half an avocado and mix with a spoonful of honey for a deeply moisturizing mask."},
            {"item": "Olive Oil & Egg Yolk Mask", "desc": "Whisk an egg yolk with a teaspoon of olive oil to nourish and hydrate."}
        ],
        "Combination": [
            {"item": "Yogurt & Honey Face Pack", "desc": "Mix yogurt with a touch of honey to balance oily and dry spots."},
            {"item": "Strawberry & Lemon Mask", "desc": "Mash a few strawberries with a squeeze of lemon juice. Apply only to oily areas like your T-zone."}
        ],
        "Normal": [
            {"item": "Green Tea & Honey Mask", "desc": "Mix brewed green tea leaves with honey to protect and hydrate your skin."},
            {"item": "Cucumber & Aloe Gel", "desc": "Blend cucumber with a bit of aloe vera gel for a refreshing and soothing treatment."}
        ]
    }

    # --- Main Logic ---
    scores = calculate_scores(data)
    skin_type = max(scores, key=scores.get)
    
    if (scores['Combination'] >= scores['Oily'] and 
        scores['Combination'] >= scores['Dry'] and
        data.get('oiliness') == 'combination'):
        skin_type = "Combination"
        
    skin_concerns = data.get('skin_concerns', [])
    is_acne_prone = any(c in skin_concerns for c in ['Acne or breakouts', 'blackheads'])
    is_aging = 'Fine lines or wrinkles' in skin_concerns
    is_sensitive = data.get('sensitivity') != 'no'
    
    products = []
    concerns = []
    advice_parts = [f"Your primary skin type is {skin_type}."]
    
    # 1. Cleanser
    if skin_type == "Oily" and is_acne_prone:
        cleansers = [p for p in all_products['cleanser']['Oily'] 
                     if 'Salicylic' in p['item'] or 'Sulfur' in p['item'] or 'Charcoal' in p['item']]
        products.append(random.choice(cleansers) if cleansers else get_random_product('cleanser', 'Oily'))
    else:
        products.append(get_random_product('cleanser', skin_type))
    
    # 2. Toner
    products.extend(get_multiple_products('toner', skin_type, 1))
    
    # 3. Treatments
    if is_acne_prone:
        products.extend(get_multiple_products('treatment', 'Acne', 2))
        concerns.append("Acne-Prone")
        advice_parts.append("Your routine focuses on controlling oil and treating breakouts.")
    
    if is_aging:
        products.extend(get_multiple_products('treatment', 'Aging', 2))
        concerns.append("Aging")
        advice_parts.append("Includes anti-aging ingredients to target fine lines.")
    
    # 4. Moisturizer
    if skin_type == "Oily":
        moisturizers = [p for p in all_products['moisturizer']['Oily'] 
                        if 'Oil-Free' in p['item'] or 'Gel' in p['item']]
        products.append(random.choice(moisturizers) if moisturizers else get_random_product('moisturizer', 'Oily'))
    else:
        products.append(get_random_product('moisturizer', skin_type))
    
    # 5. Sunscreen
    sunscreen_choices = all_products['sunscreen']['all']
    if skin_type == "Oily":
        sunscreens = [p for p in sunscreen_choices if 'Oil-Free' in p['item'] or 'Mattifying' in p['item']]
        products.append(random.choice(sunscreens) if sunscreens else random.choice(sunscreen_choices))
    else:
        products.append(random.choice(sunscreen_choices))

    # 6. Extra products
    products.extend(get_multiple_products('extra', skin_type, 1))
    
    # DIY Remedies
    diy = get_multiple_diy(skin_type, 2)
    if is_acne_prone and skin_type == "Oily":
        acne_remedies = [r for r in all_diy.get('Oily', []) if 'acne' in r['desc'] or 'blemishes' in r['desc']]
        if acne_remedies and len(diy) < 3:
            diy.append(random.choice(acne_remedies))
    
    if is_sensitive:
        concerns.append("Sensitive")
        advice_parts.append("We've selected gentle formulations to avoid irritation.")
        
    if concerns:
        skin_type += f" ({', '.join(concerns)})"
        
    return {
        "skin_type": skin_type,
        "advice": " ".join(advice_parts),
        "products": [p for p in products if p],
        "diy": diy[:3]
    }