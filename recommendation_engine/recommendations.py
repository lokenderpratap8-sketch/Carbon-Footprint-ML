import pickle

class RecommendationEngine:
    def __init__(self, classifier_metadata_path='models/classifier_metadata.pkl'):
        """Initialize recommendation engine with category mappings"""
        
        # Load classifier metadata
        with open(classifier_metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.reverse_mapping = metadata['reverse_mapping']
        
        # Category-based recommendations
        self.recommendations = {
            'Low': [
                "Maintain your current sustainable habits!",
                "Consider sharing your eco-friendly practices with others.",
                "Explore additional renewable energy options.",
                "Encourage friends and family to adopt sustainable practices."
            ],
            'Medium': [
                "Switch to LED lighting throughout your home.",
                "Reduce air conditioning usage by 2-3°C.",
                "Consider carpooling or public transport for work commutes.",
                "Try meat-free meals 2-3 times per week.",
                "Use energy-efficient appliances.",
                "Reduce online shopping frequency.",
                "Choose local and seasonal food products."
            ],
            'High': [
                "Install solar panels to reduce electricity dependency.",
                "Switch to an electric or hybrid vehicle.",
                "Reduce air travel by combining trips or using video conferencing.",
                "Adopt a plant-based diet for most meals.",
                "Improve home insulation to reduce heating/cooling costs.",
                "Use smart thermostats to optimize energy usage.",
                "Minimize fast fashion purchases.",
                "Increase recycling and waste segregation practices."
            ],
            'Critical': [
                "URGENT: Your carbon footprint is critically high.",
                "Immediately switch to renewable energy sources.",
                "Eliminate non-essential air travel.",
                "Transition to a fully plant-based diet.",
                "Replace all vehicles with electric alternatives.",
                "Invest in carbon offset programs.",
                "Consider professional energy audit for your home.",
                "Reduce overall consumption and waste significantly.",
                "Adopt zero-waste lifestyle practices."
            ]
        }
        
        # Feature-specific recommendations by category
        self.feature_recommendations = {
            # Transportation
            'Car Travel (km/month)': [
                "Use public transportation when available",
                "Carpool with colleagues or friends",
                "Switch to a hybrid or electric vehicle",
                "Combine multiple errands into one trip",
                "Consider biking or walking for short distances",
                "Maintain proper tire pressure for better fuel efficiency"
            ],
            'Fuel Type': [
                "Consider switching from petrol to diesel (more efficient)",
                "Best option: Switch to electric vehicle",
                "Maintain your vehicle properly for optimal efficiency",
                "Use eco-driving techniques (smooth acceleration/braking)"
            ],
            'Public Transport (hrs/week)': [
                "Good choice! Continue using public transport",
                "Encourage others to use public transport",
                "Consider off-peak travel to reduce congestion"
            ],
            'Train Travel (km/year)': [
                "Train travel is very eco-friendly - continue this habit",
                "Choose train over short-haul flights when possible"
            ],
            'Bike Usage (hrs/week)': [
                "Excellent! Biking is zero-emission transport",
                "Consider biking for more trips",
                "Encourage community biking initiatives"
            ],
            'Cab/Rideshare (trips/month)': [
                "Reduce cab usage by planning trips better",
                "Use public transport instead of cabs when possible",
                "Share rides with others"
            ],
            'Vehicle Type': [
                "SUVs and trucks have higher emissions - consider downsizing",
                "Hatchbacks and sedans are more fuel-efficient",
                "Electric vehicles eliminate tailpipe emissions"
            ],
            'Vehicle Age (years)': [
                "Older vehicles are less efficient - consider upgrading",
                "Newer vehicles have better emission standards",
                "Regular maintenance improves efficiency of older vehicles"
            ],
            'Number of Vehicles': [
                "Consider reducing the number of household vehicles",
                "Share vehicles among family members",
                "Use car-sharing services instead of owning multiple vehicles"
            ],
            
            # Household Energy
            'Electricity (kWh/month)': [
                "Switch to LED bulbs (75% less energy than incandescent)",
                "Use smart power strips to eliminate phantom loads",
                "Set thermostat to 68°F in winter, 78°F in summer",
                "Insulate your home properly",
                "Consider installing solar panels",
                "Unplug devices when not in use"
            ],
            'Household Members': [
                "Educate all household members about energy conservation",
                "Set family energy-saving goals",
                "Share energy-saving responsibilities"
            ],
            'AC Usage (hrs/day)': [
                "Use ceiling fans instead of AC when possible",
                "Set AC to 24-26°C instead of lower temperatures",
                "Use natural ventilation during cooler hours",
                "Maintain AC units regularly for efficiency"
            ],
            'Renewable Energy': [
                "Excellent! Renewable energy significantly reduces emissions",
                "Consider increasing renewable energy usage",
                "Share your renewable energy experience with others"
            ],
            'Solar Panels': [
                "Great! Solar panels are excellent for reducing emissions",
                "Consider adding battery storage for maximum benefit",
                "Monitor your solar production regularly"
            ],
            'LPG Consumption (kg/month)': [
                "Reduce LPG usage by cooking efficiently",
                "Consider electric cooking with renewable energy",
                "Use pressure cookers to reduce cooking time"
            ],
            'Home Type': [
                "Apartments generally have lower per-person emissions",
                "Independent houses can benefit from better insulation",
                "Consider shared walls for better thermal efficiency"
            ],
            
            # Food and Diet
            'Diet Type': [
                "Plant-based diets have the lowest carbon footprint",
                "Reducing meat consumption significantly lowers emissions",
                "Local and seasonal foods reduce transportation emissions"
            ],
            'Meat Meals (per week)': [
                "Try Meatless Mondays",
                "Replace beef with chicken or fish (lower carbon footprint)",
                "Explore plant-based protein alternatives",
                "Reduce portion sizes of meat dishes"
            ],
            'Dairy Consumption (servings/day)': [
                "Try plant-based milk alternatives",
                "Reduce cheese consumption (high emissions)",
                "Choose locally-produced dairy products"
            ],
            'Packaged Food (meals/week)': [
                "Cook fresh meals instead of packaged food",
                "Buy in bulk to reduce packaging waste",
                "Choose products with minimal packaging"
            ],
            'Food Waste (%)': [
                "Plan meals ahead to reduce waste",
                "Compost organic waste",
                "Use leftovers creatively",
                "Buy only what you need"
            ],
            'Local Food Consumption (%)': [
                "Excellent! Local food reduces transportation emissions",
                "Shop at farmers markets",
                "Grow your own vegetables when possible"
            ],
            
            # Travel
            'Flight Hours (per year)': [
                "Take trains for domestic travel when possible",
                "Use video conferencing instead of business trips",
                "Choose direct flights (takeoffs/landings use most fuel)",
                "Offset your flight emissions through certified programs",
                "Combine multiple trips into one longer visit"
            ],
            'Domestic Flights (per year)': [
                "Consider trains for domestic travel under 500km",
                "Combine multiple domestic trips",
                "Use virtual meetings instead of domestic travel"
            ],
            'International Flights (per year)': [
                "International flights have high emissions - reduce frequency",
                "Choose economy class (lower per-passenger emissions)",
                "Stay longer at destinations to reduce frequency",
                "Consider vacationing closer to home"
            ],
            'Hotel Stays (per year)': [
                "Choose eco-certified hotels",
                "Reuse towels and linens",
                "Turn off lights and AC when leaving room"
            ],
            'Vacation Frequency (trips/year)': [
                "Consider staycations or local vacations",
                "Combine work and leisure travel",
                "Choose destinations with sustainable tourism practices"
            ],
            
            # Consumer Behavior
            'Online Shopping (USD/month)': [
                "Reduce impulse online purchases",
                "Combine orders to reduce shipping",
                "Choose slower shipping options (lower emissions)",
                "Buy from local stores when possible"
            ],
            'Fast Fashion (items/month)': [
                "Fast fashion has high environmental impact - reduce purchases",
                "Buy quality clothing that lasts longer",
                "Choose sustainable and ethical fashion brands",
                "Donate or recycle old clothes instead of discarding"
            ],
            'Electronics (devices/year)': [
                "Extend device lifespan through proper care",
                "Buy refurbished electronics when possible",
                "Recycle old electronics properly",
                "Choose energy-efficient devices"
            ],
            'Recycling Score (1-10)': [
                "Excellent recycling habits - maintain this!",
                "Increase recycling of paper, plastic, and glass",
                "Learn about local recycling guidelines",
                "Start composting organic waste"
            ],
            'Waste Segregation': [
                "Great! Waste segregation reduces landfill emissions",
                "Educate others about proper waste segregation",
                "Reduce waste generation at source"
            ]
        }
        
        print("✓ Recommendation Engine initialized")
    
    def get_category_recommendations(self, category):
        """Get recommendations based on emission category"""
        if isinstance(category, int):
            category = self.reverse_mapping.get(category, 'Medium')
        
        return self.recommendations.get(category, self.recommendations['Medium'])
    
    def get_feature_specific_recommendations(self, top_contributors, max_per_feature=2):
        """Get recommendations based on top contributing features"""
        
        feature_recs = []
        
        for feature in top_contributors.keys():
            if feature in self.feature_recommendations:
                recs = self.feature_recommendations[feature][:max_per_feature]
                feature_recs.extend(recs)
        
        return feature_recs
    
    def get_personalized_recommendations(self, category, top_contributors, category_breakdown=None):
        """Get complete personalized recommendations"""
        
        # Get category-based recommendations
        category_recs = self.get_category_recommendations(category)
        
        # Get feature-specific recommendations
        feature_recs = self.get_feature_specific_recommendations(top_contributors)
        
        # Add category-specific recommendations based on breakdown
        if category_breakdown:
            category_specific_recs = self._get_category_based_recommendations(category_breakdown)
            feature_recs.extend(category_specific_recs)
        
        # Combine and deduplicate
        all_recs = category_recs + feature_recs
        unique_recs = list(dict.fromkeys(all_recs))  # Preserve order while removing duplicates
        
        return unique_recs[:12]  # Limit to top 12 recommendations
    
    def _get_category_based_recommendations(self, category_breakdown):
        """Get recommendations based on category breakdown"""
        recommendations = []
        
        # Find highest contributing category
        max_category = max(category_breakdown, key=category_breakdown.get)
        
        if max_category == 'Transportation':
            recommendations.extend([
                "Focus on reducing transportation emissions",
                "Consider switching to electric vehicles",
                "Increase use of public transport and active transport"
            ])
        elif max_category == 'Household Energy':
            recommendations.extend([
                "Focus on reducing household energy consumption",
                "Improve home insulation",
                "Switch to renewable energy sources"
            ])
        elif max_category == 'Food and Diet':
            recommendations.extend([
                "Focus on dietary changes to reduce emissions",
                "Reduce meat and dairy consumption",
                "Choose local and seasonal foods"
            ])
        elif max_category == 'Travel':
            recommendations.extend([
                "Focus on reducing travel-related emissions",
                "Reduce air travel frequency",
                "Choose eco-friendly accommodations"
            ])
        elif max_category == 'Consumer Behavior':
            recommendations.extend([
                "Focus on sustainable consumption habits",
                "Reduce fast fashion and impulse purchases",
                "Improve recycling and waste management"
            ])
        
        return recommendations
    
    def get_priority_actions(self, category, top_contributors):
        """Get priority actions based on category and top contributors"""
        
        priority_actions = []
        
        if category == 'Critical':
            priority_actions.append("🚨 IMMEDIATE ACTION REQUIRED")
        elif category == 'High':
            priority_actions.append("⚠️ HIGH PRIORITY ACTION NEEDED")
        
        # Add top 3 priority actions based on top contributors
        for i, (feature, contribution) in enumerate(list(top_contributors.items())[:3]):
            if contribution > 20:
                priority_actions.append(f"🔴 CRITICAL: Reduce {feature}")
            elif contribution > 15:
                priority_actions.append(f"🟠 HIGH: Optimize {feature}")
            elif contribution > 10:
                priority_actions.append(f"🟡 MEDIUM: Improve {feature}")
            else:
                priority_actions.append(f"🟢 GOOD: Maintain {feature}")
        
        return priority_actions

if __name__ == "__main__":
    print("Testing Recommendation Engine...")
    
    # Initialize
    engine = RecommendationEngine()
    
    # Test category recommendations
    print("\n" + "="*50)
    print("CATEGORY RECOMMENDATIONS")
    print("="*50)
    
    for category in ['Low', 'Medium', 'High', 'Critical']:
        print(f"\n{category}:")
        for rec in engine.get_category_recommendations(category)[:3]:
            print(f"  • {rec}")
    
    # Test feature-specific recommendations
    print("\n" + "="*50)
    print("FEATURE-SPECIFIC RECOMMENDATIONS")
    print("="*50)
    
    top_contributors = {
        'Annual Energy Consumption': 23.0,
        'AC Usage (hrs/day)': 13.3,
        'Renewable Energy': 10.3
    }
    
    feature_recs = engine.get_feature_specific_recommendations(top_contributors)
    for rec in feature_recs[:6]:
        print(f"  • {rec}")
    
    # Test personalized recommendations
    print("\n" + "="*50)
    print("PERSONALIZED RECOMMENDATIONS")
    print("="*50)
    
    category_breakdown = {
        'Transportation': 74.7,
        'Household Energy': 20.1,
        'Food and Diet': 0.7,
        'Travel': 1.4,
        'Consumer Behavior': 3.2
    }
    
    personalized = engine.get_personalized_recommendations('High', top_contributors, category_breakdown)
    for rec in personalized[:8]:
        print(f"  • {rec}")
    
    # Test priority actions
    print("\n" + "="*50)
    print("PRIORITY ACTIONS")
    print("="*50)
    
    priorities = engine.get_priority_actions('High', top_contributors)
    for action in priorities:
        print(f"  {action}")
    
    print("\n✓ Recommendation Engine test completed successfully!")
