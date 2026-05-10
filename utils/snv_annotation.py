import pandas as pd
import argparse

def preprocess_vcf_annotations(input_file, output_file):
    df = pd.read_csv(input_file, header=0, low_memory=False, encoding='latin1')

    # 1. Filter for specific samples
    target_samples = ['K4B', 'K5B', 'K6B', 'K7B', 'K8B']
    df = df[df['sample'].isin(target_samples)].copy()

    # 2. Create ch_location: ch<i>_<loc>
    chromosome = df['CHROM'].astype(str)
    chromosome = chromosome.replace('99', 'X')
    df['ch_location'] = 'chr' + chromosome + '_' + df['POS'].astype(str)

    # 3. Logic for location_type and location
    def identify_location(row):
        is_exon = str(row.get('inExon')).upper() == 'TRUE'
        is_intron = str(row.get('inIntron')).upper() == 'TRUE'
        consequence = str(row.get('Consequence')).lower()
        
        # Check Exon/Intron first
        if is_exon:
            return 'exon', row.get('EXON')
        elif is_intron:
            return 'intron', row.get('INTRON')
        
        # 4. New Logic: Check Consequence for Upstream/Downstream
        elif 'upstream' in consequence:
            return 'upstream', None
        elif 'downstream' in consequence:
            return 'downstream', None
        else:
            return 'other', None

    loc_data = df.apply(identify_location, axis=1)
    df['location_type'] = [x[0] for x in loc_data]
    df['location'] = [x[1] for x in loc_data]

    # Mapping and column selection
    column_mapping = {
        'sample': 'sample_id',
        'type': 'type',
        'REF': 'ref_Allele',
        'ALT': 'alt_Allele',
        'SYMBOL': 'gene',
        'Pathogenicity': 'pathogenic',
        'Consequence': 'consequence'
    }

    final_columns = [
        'sample_id', 'ch_location', 'type', 'ref_Allele', 'alt_Allele', 
        'location_type', 'location', 'gene', 'pathogenic', 'consequence'
    ]

    processed_df = df.rename(columns=column_mapping)[final_columns].sort_values(by=['sample_id', 'ch_location']).reset_index(drop=True)
    processed_df.to_csv(output_file, index=False)
    
    return processed_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess VCF annotations for SNV metadata.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file containing VCF annotations.')
    parser.add_argument('output_file', type=str, help='Path to the output CSV file for processed SNV metadata.')

    args = parser.parse_args()
    preprocess_vcf_annotations(args.input_file, args.output_file)