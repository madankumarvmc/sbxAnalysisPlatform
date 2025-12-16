"""
Prompt Templates for Word Report Generation

This module contains all prompt templates for different sections of the Word report.
Each section can have multiple prompts for different data views.
"""

PROMPTS = {
    'executive_summary': {
        'main': """You are a warehouse operations analyst. Analyze this comprehensive warehouse data and provide an executive summary.

Data Overview:
{all_metrics}

Please provide a 2-3 paragraph executive summary that includes:
1. Overall operational performance highlights
2. Key trends and patterns observed
3. Top 3-5 actionable insights for management

Focus on business impact and strategic recommendations. Use professional business language."""
    },
    
    'receipt_analysis': {
        'daily_summary': """You are a warehouse receiving operations expert. Analyze these daily receipt patterns:

{daily_data}

Please provide insights covering:
1. Volume trends and patterns (daily/weekly variations)
2. Peak receiving periods and capacity implications
3. Any anomalies or concerns in the receiving pattern
4. Operational efficiency observations

Keep the analysis concise (2-3 paragraphs) and focus on actionable insights.""",

        'percentile_analysis': """Review these receipt volume percentiles for capacity planning:

{percentile_data}

Explain:
1. What these percentiles mean for staffing requirements
2. Capacity planning recommendations
3. Risk areas if volumes exceed certain percentiles

Provide practical recommendations in 1-2 paragraphs.""",

        'overall': """Based on the complete receipt analysis data:

{summary_data}

Provide:
1. Overall assessment of receiving operations
2. Top 3 recommendations for improving receiving efficiency
3. Any risks or concerns that need immediate attention

Keep recommendations specific and actionable (2 paragraphs)."""
    },
    
    'order_analysis': {
        'daily_patterns': """Analyze these daily order patterns and volumes:

{daily_data}

Identify:
1. Order volume trends and seasonality
2. Peak periods and their impact
3. Fulfillment capacity requirements
4. Any concerning patterns

Provide insights in 2-3 paragraphs with focus on operations planning.""",

        'percentile_analysis': """Review order volume percentiles:

{percentile_data}

Explain implications for:
1. Warehouse staffing levels
2. Peak capacity management
3. Service level maintenance

Keep analysis practical and focused (1-2 paragraphs).""",

        'sku_distribution': """Analyze this SKU order distribution:

{sku_data}

Comment on:
1. SKU velocity patterns
2. Storage strategy implications
3. Picking efficiency opportunities

Provide actionable insights in 2 paragraphs."""
    },
    
    'sku_analysis': {
        'performance': """Analyze SKU performance metrics:

{sku_metrics}

Provide insights on:
1. Top performing SKUs and their characteristics
2. Slow-moving items requiring attention
3. Inventory optimization opportunities

Focus on inventory management implications (2-3 paragraphs).""",

        'abc_classification': """Review this ABC classification data:

{abc_data}

Explain:
1. Distribution insights across A, B, C categories
2. Storage and handling strategy recommendations
3. Inventory investment optimization

Provide strategic recommendations in 2 paragraphs."""
    },
    
    'inventory_analysis': {
        'stock_levels': """Analyze current inventory positions:

{inventory_data}

Assess:
1. Stock health and aging concerns
2. Overstock/understock situations
3. Working capital implications
4. Turnover optimization opportunities

Provide analysis in 2-3 paragraphs focusing on financial impact.""",

        'recommendations': """Based on inventory analysis:

{summary_data}

Provide:
1. Top 3 inventory optimization actions
2. Risk mitigation strategies
3. Expected benefits of recommendations

Keep recommendations specific and measurable (2 paragraphs)."""
    },
    
    'manpower_analysis': {
        'picking_requirements': """Analyze picking manpower requirements:

{picking_data}

Provide insights on:
1. Current vs. required staffing levels
2. Efficiency improvement opportunities
3. Peak period staffing strategies

Focus on operational feasibility (2 paragraphs).""",

        'receiving_requirements': """Analyze receiving manpower needs:

{receiving_data}

Comment on:
1. Receiving team sizing adequacy
2. Productivity improvement areas
3. Cross-training opportunities

Provide practical recommendations (1-2 paragraphs).""",

        'overall_workforce': """Review overall workforce analysis:

{workforce_data}

Synthesize:
1. Total manpower optimization potential
2. Skill gap areas
3. Cost-benefit of staffing changes

Provide strategic workforce recommendations (2-3 paragraphs)."""
    }
}

FALLBACKS = {
    'executive_summary': """Based on the warehouse operational data analysis, the facility demonstrates standard operational patterns with opportunities for optimization in key areas. Volume trends indicate stable operations with predictable peak periods requiring focused resource allocation.

Key recommendations include enhancing inventory turnover, optimizing picking routes, and implementing dynamic staffing models to better align with demand patterns. These improvements could yield significant efficiency gains and cost reductions.""",
    
    'receipt_analysis': """Receipt analysis indicates regular inbound flow patterns with manageable volume variations. Daily receiving operations show consistent performance with identifiable peak periods that align with typical supply chain patterns.

Opportunities exist to optimize receiving dock scheduling and improve putaway efficiency through better resource allocation during peak periods.""",
    
    'order_analysis': """Order patterns demonstrate consistent daily volumes with expected weekly variations. The operation maintains stable fulfillment rates with opportunities to improve efficiency through better wave planning and resource allocation.""",
    
    'sku_analysis': """SKU performance analysis reveals typical velocity distribution with clear fast, medium, and slow-moving segments. Optimization opportunities exist in storage location strategies and inventory level adjustments.""",
    
    'inventory_analysis': """Inventory levels show standard patterns with opportunities to improve turnover rates and reduce carrying costs through better demand planning and stock optimization strategies.""",
    
    'manpower_analysis': """Workforce analysis indicates current staffing levels are generally aligned with operational requirements, with specific opportunities for productivity improvements through training and process optimization.""",
    
    'general': """Analysis completed based on available warehouse operational data. Results indicate standard performance patterns with identified opportunities for improvement in efficiency and cost optimization."""
}