"""
Cost analysis script for comparing ChromaDB vs Managed Vector DBs.
Analyzes costs for different vector scales: 100K, 1M, 10M vectors.
"""

import csv
from pathlib import Path
from typing import Dict, List


class CostAnalyzer:
    """Analyzes and compares costs between ChromaDB and managed vector databases."""
    
    def __init__(self):
        """Initialize cost analyzer with pricing data."""
        # Pricing estimates (USD per month)
        # These are approximate prices based on common managed vector DB providers
        self.managed_db_pricing = {
            'pinecone': {
                'starter': {
                    'vectors': 100000,
                    'price': 70.0,
                    'notes': 'Starter tier, 1GB storage'
                },
                'production': {
                    'vectors': 1000000,
                    'price': 700.0,
                    'notes': 'Production tier, 10GB storage'
                },
                'enterprise': {
                    'vectors': 10000000,
                    'price': 7000.0,
                    'notes': 'Enterprise tier, 100GB storage'
                }
            },
            'weaviate_cloud': {
                'starter': {
                    'vectors': 100000,
                    'price': 50.0,
                    'notes': 'Sandbox tier, 1GB storage'
                },
                'production': {
                    'vectors': 1000000,
                    'price': 500.0,
                    'notes': 'Production tier, 10GB storage'
                },
                'enterprise': {
                    'vectors': 10000000,
                    'price': 5000.0,
                    'notes': 'Enterprise tier, 100GB storage'
                }
            },
            'qdrant_cloud': {
                'starter': {
                    'vectors': 100000,
                    'price': 40.0,
                    'notes': 'Community tier, 1GB storage'
                },
                'production': {
                    'vectors': 1000000,
                    'price': 400.0,
                    'notes': 'Production tier, 10GB storage'
                },
                'enterprise': {
                    'vectors': 10000000,
                    'price': 4000.0,
                    'notes': 'Enterprise tier, 100GB storage'
                }
            }
        }
        
        # ChromaDB costs (self-hosted)
        # Based on typical cloud instance costs
        self.chroma_pricing = {
            '100k': {
                'vectors': 100000,
                'instance_cost': 20.0,  # Small instance
                'storage_cost': 5.0,    # 1GB storage
                'bandwidth_cost': 10.0, # Estimated
                'maintenance_cost': 20.0, # Dev time estimate
                'total': 55.0
            },
            '1m': {
                'vectors': 1000000,
                'instance_cost': 80.0,  # Medium instance
                'storage_cost': 20.0,   # 10GB storage
                'bandwidth_cost': 40.0, # Estimated
                'maintenance_cost': 80.0, # Dev time estimate
                'total': 220.0
            },
            '10m': {
                'vectors': 10000000,
                'instance_cost': 320.0, # Large instance
                'storage_cost': 80.0,   # 100GB storage
                'bandwidth_cost': 160.0, # Estimated
                'maintenance_cost': 320.0, # Dev time estimate
                'total': 880.0
            }
        }
    
    def calculate_chroma_costs(self) -> List[Dict]:
        """Calculate ChromaDB costs for different scales."""
        results = []
        
        for scale, data in self.chroma_pricing.items():
            results.append({
                'solution': 'ChromaDB (Self-Hosted)',
                'scale': scale,
                'vectors': data['vectors'],
                'instance_cost': data['instance_cost'],
                'storage_cost': data['storage_cost'],
                'bandwidth_cost': data['bandwidth_cost'],
                'maintenance_cost': data['maintenance_cost'],
                'total_monthly': data['total'],
                'annual_cost': data['total'] * 12
            })
        
        return results
    
    def calculate_managed_costs(self) -> List[Dict]:
        """Calculate managed vector DB costs for different scales."""
        results = []
        
        for provider, tiers in self.managed_db_pricing.items():
            for tier_name, data in tiers.items():
                results.append({
                    'solution': f'{provider.capitalize()} ({tier_name.capitalize()})',
                    'scale': self._get_scale_name(data['vectors']),
                    'vectors': data['vectors'],
                    'instance_cost': 0.0,  # Included in managed pricing
                    'storage_cost': 0.0,  # Included in managed pricing
                    'bandwidth_cost': 0.0,  # Included in managed pricing
                    'maintenance_cost': 0.0,  # Included in managed pricing
                    'total_monthly': data['price'],
                    'annual_cost': data['price'] * 12,
                    'notes': data['notes']
                })
        
        return results
    
    def _get_scale_name(self, vectors: int) -> str:
        """Get scale name based on vector count."""
        if vectors >= 10000000:
            return '10m'
        elif vectors >= 1000000:
            return '1m'
        else:
            return '100k'
    
    def compare_costs(self) -> List[Dict]:
        """Compare costs between ChromaDB and managed solutions."""
        chroma_costs = self.calculate_chroma_costs()
        managed_costs = self.calculate_managed_costs()
        
        all_costs = chroma_costs + managed_costs
        
        # Sort by scale and total cost
        scale_order = {'100k': 1, '1m': 2, '10m': 3}
        all_costs.sort(key=lambda x: (scale_order.get(x['scale'], 99), x['total_monthly']))
        
        return all_costs
    
    def generate_savings_analysis(self) -> Dict:
        """Generate savings analysis comparing ChromaDB to managed solutions."""
        chroma_costs = self.calculate_chroma_costs()
        managed_costs = self.calculate_managed_costs()
        
        savings = {}
        
        for chroma in chroma_costs:
            scale = chroma['scale']
            scale_managed = [m for m in managed_costs if m['scale'] == scale]
            
            if scale_managed:
                avg_managed = sum(m['total_monthly'] for m in scale_managed) / len(scale_managed)
                monthly_savings = avg_managed - chroma['total_monthly']
                annual_savings = monthly_savings * 12
                savings_percentage = (monthly_savings / avg_managed) * 100
                
                savings[scale] = {
                    'chroma_monthly': chroma['total_monthly'],
                    'avg_managed_monthly': avg_managed,
                    'monthly_savings': monthly_savings,
                    'annual_savings': annual_savings,
                    'savings_percentage': savings_percentage
                }
        
        return savings
    
    def export_to_csv(self, output_file: Path = None) -> Path:
        """Export cost comparison to CSV file."""
        if output_file is None:
            output_file = Path("cost_comparison.csv")
        
        costs = self.compare_costs()
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'solution', 'scale', 'vectors', 'instance_cost',
                'storage_cost', 'bandwidth_cost', 'maintenance_cost',
                'total_monthly', 'annual_cost', 'notes'
            ])
            writer.writeheader()
            
            for cost in costs:
                writer.writerow(cost)
        
        return output_file
    
    def print_comparison(self):
        """Print cost comparison to console."""
        print("\n" + "="*80)
        print("COST ANALYSIS: ChromaDB vs Managed Vector Databases")
        print("="*80)
        
        costs = self.compare_costs()
        
        current_scale = None
        for cost in costs:
            if cost['scale'] != current_scale:
                current_scale = cost['scale']
                print(f"\n{'='*80}")
                print(f"Scale: {current_scale.upper()} ({cost['vectors']:,} vectors)")
                print("="*80)
            
            print(f"\n{cost['solution']}")
            print(f"  Monthly Cost: ${cost['total_monthly']:.2f}")
            print(f"  Annual Cost:  ${cost['annual_cost']:.2f}")
            
            if cost['instance_cost'] > 0:
                print(f"  Breakdown:")
                print(f"    - Instance:    ${cost['instance_cost']:.2f}")
                print(f"    - Storage:     ${cost['storage_cost']:.2f}")
                print(f"    - Bandwidth:   ${cost['bandwidth_cost']:.2f}")
                print(f"    - Maintenance: ${cost['maintenance_cost']:.2f}")
            
            if 'notes' in cost:
                print(f"  Notes: {cost['notes']}")
        
        # Print savings analysis
        savings = self.generate_savings_analysis()
        
        print("\n" + "="*80)
        print("SAVINGS ANALYSIS (ChromaDB vs Average Managed)")
        print("="*80)
        
        for scale, data in savings.items():
            print(f"\nScale: {scale.upper()}")
            print(f"  ChromaDB Monthly:     ${data['chroma_monthly']:.2f}")
            print(f"  Managed Monthly:      ${data['avg_managed_monthly']:.2f}")
            print(f"  Monthly Savings:      ${data['monthly_savings']:.2f}")
            print(f"  Annual Savings:       ${data['annual_savings']:.2f}")
            print(f"  Savings Percentage:   {data['savings_percentage']:.1f}%")
        
        print("\n" + "="*80)
        print("TRADEOFFS AND CONSIDERATIONS")
        print("="*80)
        print("""
ChromaDB (Self-Hosted):
  Pros:
    - Significant cost savings (60-80% cheaper)
    - Full control over data and infrastructure
    - No vendor lock-in
    - Customizable to specific needs
    - Privacy and data sovereignty
  
  Cons:
    - Requires DevOps expertise
    - Maintenance overhead
    - Need to handle scaling manually
    - No built-in high availability (need to configure)
    - Monitoring and alerting setup required

Managed Vector Databases:
  Pros:
    - Zero infrastructure management
    - Built-in high availability
    - Automatic scaling
    - Managed backups and updates
    - 24/7 support
    - Better for teams without DevOps
  
  Cons:
    - Higher costs (3-5x more expensive)
    - Vendor lock-in
    - Less control over configuration
    - Data residency concerns
    - Usage-based pricing can be unpredictable

WHEN TO SWITCH TO MANAGED DB:
  - When team lacks DevOps expertise
  - When requiring 99.99%+ uptime SLA
  - When scaling beyond 10M vectors
  - When compliance requires managed services
  - When maintenance cost exceeds managed pricing
  - When needing global edge deployment
        """)
        print("="*80 + "\n")


def main():
    """Main function to run cost analysis."""
    analyzer = CostAnalyzer()
    
    # Print comparison
    analyzer.print_comparison()
    
    # Export to CSV
    csv_file = analyzer.export_to_csv()
    print(f"\nCost comparison exported to: {csv_file}")


if __name__ == "__main__":
    main()
