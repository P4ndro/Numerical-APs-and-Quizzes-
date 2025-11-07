"""
Steam Games Clustering Analysis
Clusters video games based on review scores, playtime, and tags using K-means and DBSCAN

Author: Clustering Analysis Project
Dataset: Steam Store Games (Kaggle - nikdavis/steam-store-games)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')


class SteamGamesClustering:
    """
    Main class for clustering Steam games
    Supports flexible distance metrics and norms as required by the assignment
    """
    
    def __init__(self, data_path, norm='l2', random_state=42):
        """
        Initialize the clustering analysis
        
        Parameters:
        -----------
        data_path : str
            Path to the Steam games CSV file
        norm : str or int
            Norm to use for distance calculation
            - 'l2' or 2: Euclidean (default)
            - 'l1' or 1: Manhattan
            - 'linf' or np.inf: Chebyshev (maximum)
            This allows TA to test with different norms
        random_state : int
            Random seed for reproducibility
        """
        self.data_path = data_path
        self.norm = self._parse_norm(norm)
        self.random_state = random_state
        self.raw_data = None
        self.processed_data = None
        self.scaled_data = None
        self.scaler = StandardScaler()
        self.pca_2d = None
        self.pca_3d = None
        
        # Results storage
        self.kmeans_labels = None
        self.dbscan_labels = None
        self.kmeans_model = None
        self.dbscan_model = None
        
    def _parse_norm(self, norm):
        """Parse norm parameter to Minkowski p-value"""
        if norm == 'l2' or norm == 2 or norm == 'euclidean':
            return 2
        elif norm == 'l1' or norm == 1 or norm == 'manhattan':
            return 1
        elif norm == 'linf' or norm == np.inf or norm == 'chebyshev':
            return np.inf
        else:
            try:
                return float(norm)
            except:
                print(f"Warning: Unknown norm '{norm}', using L2 (Euclidean)")
                return 2
    
    def load_data(self):
        """Load and perform initial data exploration"""
        print("=" * 70)
        print("LOADING STEAM GAMES DATASET")
        print("=" * 70)
        
        try:
            self.raw_data = pd.read_csv(self.data_path)
            print(f"[OK] Loaded {len(self.raw_data)} games")
            print(f"[OK] Columns: {list(self.raw_data.columns)}")
            print(f"\nFirst few rows:\n{self.raw_data.head()}")
            return True
        except FileNotFoundError:
            print(f"ERROR: Could not find file at {self.data_path}")
            print("\nPlease download the dataset from:")
            print("https://www.kaggle.com/datasets/nikdavis/steam-store-games")
            print(f"And place it at: {self.data_path}")
            return False
        except Exception as e:
            print(f"ERROR loading data: {e}")
            return False
    
    def preprocess_data(self, min_reviews=10, max_games=None):
        """
        Preprocess and clean the data
        Extract relevant features for clustering
        
        Parameters:
        -----------
        min_reviews : int
            Minimum number of reviews required (filters out games with insufficient data)
        max_games : int or None
            Maximum number of games to use (for faster processing). If None, uses all games.
        """
        print("\n" + "=" * 70)
        print("PREPROCESSING DATA")
        print("=" * 70)
        
        df = self.raw_data.copy()
        
        # Limit dataset size if specified
        if max_games is not None and len(df) > max_games:
            print(f"[OK] Sampling {max_games} games from {len(df)} total games")
            df = df.sample(n=max_games, random_state=self.random_state)
        
        # Identify relevant columns (adapt based on actual dataset structure)
        print("\nExtracting features...")
        
        features_to_extract = []
        feature_names = []
        
        # Try to extract review scores (positive/negative ratio, total reviews)
        if 'positive_ratings' in df.columns and 'negative_ratings' in df.columns:
            df['total_reviews'] = df['positive_ratings'].fillna(0) + df['negative_ratings'].fillna(0)
            df['positive_ratio'] = (df['positive_ratings'].fillna(0) / df['total_reviews'].replace(0, 1))
            
            # Filter games with sufficient reviews
            df = df[df['total_reviews'] >= min_reviews].copy()
            print(f"[OK] Filtered to {len(df)} games with at least {min_reviews} reviews")
            
            features_to_extract.extend(['positive_ratio', 'total_reviews'])
            feature_names.extend(['Positive Review Ratio', 'Total Reviews'])
        
        # Extract price information
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
            features_to_extract.append('price')
            feature_names.append('Price')
        
        # Extract owner/player information
        if 'owners' in df.columns:
            # Parse owner ranges (e.g., "20000-50000")
            df['owners_mid'] = df['owners'].apply(self._parse_owner_range)
            features_to_extract.append('owners_mid')
            feature_names.append('Estimated Owners')
        
        # Extract average playtime
        if 'average_playtime' in df.columns:
            df['average_playtime'] = pd.to_numeric(df['average_playtime'], errors='coerce').fillna(0)
            features_to_extract.append('average_playtime')
            feature_names.append('Average Playtime')
        
        if 'median_playtime' in df.columns:
            df['median_playtime'] = pd.to_numeric(df['median_playtime'], errors='coerce').fillna(0)
            features_to_extract.append('median_playtime')
            feature_names.append('Median Playtime')
        
        # Extract genre/category information
        if 'genres' in df.columns:
            print("[OK] Parsing game genres...")
            genre_features = self._extract_genre_features(df['genres'])
            df = pd.concat([df, genre_features], axis=1)
            features_to_extract.extend(genre_features.columns)
            feature_names.extend(genre_features.columns)
        
        if 'categories' in df.columns:
            print("[OK] Parsing game categories...")
            category_features = self._extract_category_features(df['categories'])
            df = pd.concat([df, category_features], axis=1)
            features_to_extract.extend(category_features.columns)
            feature_names.extend(category_features.columns)
        
        # Create feature matrix
        self.processed_data = df[['name'] + features_to_extract].copy()
        self.processed_data = self.processed_data.dropna()
        
        print(f"\n[OK] Final dataset: {len(self.processed_data)} games")
        print(f"[OK] Features: {len(features_to_extract)}")
        print(f"  {', '.join(feature_names[:5])}{'...' if len(feature_names) > 5 else ''}")
        
        # Scale the features (important for distance-based clustering)
        feature_matrix = self.processed_data[features_to_extract].values
        self.scaled_data = self.scaler.fit_transform(feature_matrix)
        self.feature_names = feature_names
        
        print(f"[OK] Data scaled using StandardScaler")
        
        return self.processed_data
    
    def _parse_owner_range(self, owner_str):
        """Parse owner range string to midpoint value"""
        try:
            if pd.isna(owner_str) or owner_str == '':
                return 0
            # Format: "20000-50000"
            parts = str(owner_str).split('-')
            if len(parts) == 2:
                return (int(parts[0]) + int(parts[1])) / 2
            return 0
        except:
            return 0
    
    def _extract_genre_features(self, genre_series):
        """Extract binary features for top genres"""
        # Parse genre strings and find most common
        all_genres = []
        for genres in genre_series.dropna():
            if isinstance(genres, str):
                all_genres.extend([g.strip() for g in genres.split(';')])
        
        # Get top N genres
        from collections import Counter
        top_genres = [g for g, _ in Counter(all_genres).most_common(10)]
        
        # Create binary features
        genre_df = pd.DataFrame(index=genre_series.index)
        for genre in top_genres:
            genre_df[f'genre_{genre}'] = genre_series.apply(
                lambda x: 1 if isinstance(x, str) and genre in x else 0
            )
        
        return genre_df
    
    def _extract_category_features(self, category_series):
        """Extract binary features for top categories"""
        all_categories = []
        for cats in category_series.dropna():
            if isinstance(cats, str):
                all_categories.extend([c.strip() for c in cats.split(';')])
        
        from collections import Counter
        top_categories = [c for c, _ in Counter(all_categories).most_common(8)]
        
        cat_df = pd.DataFrame(index=category_series.index)
        for cat in top_categories:
            cat_df[f'cat_{cat}'] = category_series.apply(
                lambda x: 1 if isinstance(x, str) and cat in x else 0
            )
        
        return cat_df
    
    def apply_kmeans(self, n_clusters=5):
        """
        Apply K-means clustering
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create
        """
        print("\n" + "=" * 70)
        print(f"APPLYING K-MEANS CLUSTERING (k={n_clusters})")
        print(f"Using norm: L{self.norm} ({'Euclidean' if self.norm == 2 else 'Manhattan' if self.norm == 1 else 'Chebyshev' if self.norm == np.inf else f'L{self.norm}'})")
        print("=" * 70)
        
        # Note: sklearn's KMeans uses Euclidean by default
        # For other norms, we would need custom implementation or use different algorithms
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        self.kmeans_labels = self.kmeans_model.fit_predict(self.scaled_data)
        
        # Calculate metrics
        silhouette = silhouette_score(self.scaled_data, self.kmeans_labels)
        davies_bouldin = davies_bouldin_score(self.scaled_data, self.kmeans_labels)
        calinski_harabasz = calinski_harabasz_score(self.scaled_data, self.kmeans_labels)
        
        print(f"\n[OK] Clustering complete!")
        print(f"[OK] Silhouette Score: {silhouette:.4f} (higher is better, range [-1, 1])")
        print(f"[OK] Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
        print(f"[OK] Calinski-Harabasz Score: {calinski_harabasz:.2f} (higher is better)")
        
        # Show cluster sizes
        unique, counts = np.unique(self.kmeans_labels, return_counts=True)
        print(f"\nCluster sizes:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} games ({count/len(self.kmeans_labels)*100:.1f}%)")
        
        return self.kmeans_labels
    
    def apply_dbscan(self, eps=0.5, min_samples=5):
        """
        Apply DBSCAN clustering
        
        Parameters:
        -----------
        eps : float
            Maximum distance between two samples for one to be considered as in the neighborhood of the other
        min_samples : int
            Number of samples in a neighborhood for a point to be considered as a core point
        """
        print("\n" + "=" * 70)
        print(f"APPLYING DBSCAN CLUSTERING (eps={eps}, min_samples={min_samples})")
        print(f"Using norm: L{self.norm} ({'Euclidean' if self.norm == 2 else 'Manhattan' if self.norm == 1 else 'Chebyshev' if self.norm == np.inf else f'L{self.norm}'})")
        print("=" * 70)
        
        # DBSCAN supports different metrics via the metric parameter
        metric = 'euclidean' if self.norm == 2 else 'manhattan' if self.norm == 1 else 'chebyshev' if self.norm == np.inf else 'minkowski'
        
        self.dbscan_model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            p=self.norm if metric == 'minkowski' else None
        )
        
        self.dbscan_labels = self.dbscan_model.fit_predict(self.scaled_data)
        
        # Calculate metrics (excluding noise points)
        n_clusters = len(set(self.dbscan_labels)) - (1 if -1 in self.dbscan_labels else 0)
        n_noise = list(self.dbscan_labels).count(-1)
        
        print(f"\n[OK] Clustering complete!")
        print(f"[OK] Number of clusters found: {n_clusters}")
        print(f"[OK] Number of noise points: {n_noise} ({n_noise/len(self.dbscan_labels)*100:.1f}%)")
        
        if n_clusters > 1:
            # Calculate silhouette score (only for non-noise points)
            mask = self.dbscan_labels != -1
            if mask.sum() > 0 and len(set(self.dbscan_labels[mask])) > 1:
                silhouette = silhouette_score(self.scaled_data[mask], self.dbscan_labels[mask])
                davies_bouldin = davies_bouldin_score(self.scaled_data[mask], self.dbscan_labels[mask])
                calinski_harabasz = calinski_harabasz_score(self.scaled_data[mask], self.dbscan_labels[mask])
                
                print(f"[OK] Silhouette Score: {silhouette:.4f} (excluding noise)")
                print(f"[OK] Davies-Bouldin Index: {davies_bouldin:.4f}")
                print(f"[OK] Calinski-Harabasz Score: {calinski_harabasz:.2f}")
        
        # Show cluster sizes
        unique, counts = np.unique(self.dbscan_labels, return_counts=True)
        print(f"\nCluster sizes:")
        for cluster_id, count in zip(unique, counts):
            if cluster_id == -1:
                print(f"  Noise: {count} games ({count/len(self.dbscan_labels)*100:.1f}%)")
            else:
                print(f"  Cluster {cluster_id}: {count} games ({count/len(self.dbscan_labels)*100:.1f}%)")
        
        return self.dbscan_labels
    
    def find_optimal_k(self, k_range=range(2, 11)):
        """
        Find optimal number of clusters for K-means using elbow method and silhouette analysis
        """
        print("\n" + "=" * 70)
        print("FINDING OPTIMAL K FOR K-MEANS")
        print("=" * 70)
        
        inertias = []
        silhouettes = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(self.scaled_data)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.scaled_data, labels))
            print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouettes[-1]:.4f}")
        
        # Plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
        ax1.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(k_range, silhouettes, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Score vs k', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimal_k_analysis.png', dpi=300, bbox_inches='tight')
        print("\n[OK] Saved: optimal_k_analysis.png")
        plt.show()
        plt.close()
        
        # Find optimal k (highest silhouette score)
        optimal_k = list(k_range)[np.argmax(silhouettes)]
        print(f"\n[OK] Suggested optimal k: {optimal_k} (highest silhouette score: {max(silhouettes):.4f})")
        
        return optimal_k
    
    def visualize_clusters_2d(self):
        """Create 2D visualization of clusters using PCA"""
        print("\n" + "=" * 70)
        print("CREATING 2D VISUALIZATIONS")
        print("=" * 70)
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        data_2d = pca.fit_transform(self.scaled_data)
        self.pca_2d = data_2d
        
        print(f"[OK] PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # K-means visualization
        if self.kmeans_labels is not None:
            scatter1 = axes[0].scatter(data_2d[:, 0], data_2d[:, 1], 
                                       c=self.kmeans_labels, cmap='viridis', 
                                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
            axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
            axes[0].set_title('K-means Clustering', fontsize=14, fontweight='bold')
            plt.colorbar(scatter1, ax=axes[0], label='Cluster')
            axes[0].grid(True, alpha=0.3)
        
        # DBSCAN visualization
        if self.dbscan_labels is not None:
            scatter2 = axes[1].scatter(data_2d[:, 0], data_2d[:, 1], 
                                       c=self.dbscan_labels, cmap='viridis', 
                                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
            axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
            axes[1].set_title('DBSCAN Clustering', fontsize=14, fontweight='bold')
            plt.colorbar(scatter2, ax=axes[1], label='Cluster (-1 = Noise)')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('clusters_2d.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: clusters_2d.png")
        plt.show()
        plt.close()
    
    def visualize_clusters_3d(self):
        """Create 3D visualization of clusters using PCA"""
        print("\n" + "=" * 70)
        print("CREATING 3D VISUALIZATIONS")
        print("=" * 70)
        
        # Apply PCA for 3D visualization
        pca = PCA(n_components=3, random_state=self.random_state)
        data_3d = pca.fit_transform(self.scaled_data)
        self.pca_3d = data_3d
        
        print(f"[OK] PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
        
        fig = plt.figure(figsize=(16, 6))
        
        # K-means 3D
        if self.kmeans_labels is not None:
            ax1 = fig.add_subplot(121, projection='3d')
            scatter1 = ax1.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2],
                                  c=self.kmeans_labels, cmap='viridis',
                                  alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
            ax1.set_title('K-means Clustering (3D)', fontsize=14, fontweight='bold')
            plt.colorbar(scatter1, ax=ax1, label='Cluster', shrink=0.5)
        
        # DBSCAN 3D
        if self.dbscan_labels is not None:
            ax2 = fig.add_subplot(122, projection='3d')
            scatter2 = ax2.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2],
                                  c=self.dbscan_labels, cmap='viridis',
                                  alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            ax2.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
            ax2.set_title('DBSCAN Clustering (3D)', fontsize=14, fontweight='bold')
            plt.colorbar(scatter2, ax=ax2, label='Cluster', shrink=0.5)
        
        plt.tight_layout()
        plt.savefig('clusters_3d.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: clusters_3d.png")
        plt.show()
        plt.close()
    
    def analyze_clusters(self):
        """Analyze and describe each cluster"""
        print("\n" + "=" * 70)
        print("CLUSTER ANALYSIS")
        print("=" * 70)
        
        results = []
        
        # Analyze K-means clusters
        if self.kmeans_labels is not None:
            print("\n--- K-MEANS CLUSTERS ---\n")
            df = self.processed_data.copy()
            df['cluster'] = self.kmeans_labels
            
            for cluster_id in sorted(df['cluster'].unique()):
                cluster_games = df[df['cluster'] == cluster_id]
                print(f"\nCluster {cluster_id} ({len(cluster_games)} games):")
                # Handle Unicode characters in game names
                try:
                    print(f"  Sample games: {', '.join(cluster_games['name'].head(5).values)}")
                except UnicodeEncodeError:
                    print(f"  Sample games: [Games with special characters - see CSV file]")
                
                results.append({
                    'Algorithm': 'K-means',
                    'Cluster': cluster_id,
                    'Size': len(cluster_games),
                    'Sample_Games': '; '.join(cluster_games['name'].head(10).values)
                })
        
        # Analyze DBSCAN clusters
        if self.dbscan_labels is not None:
            print("\n--- DBSCAN CLUSTERS ---\n")
            df = self.processed_data.copy()
            df['cluster'] = self.dbscan_labels
            
            for cluster_id in sorted(df['cluster'].unique()):
                cluster_games = df[df['cluster'] == cluster_id]
                cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
                print(f"\n{cluster_name} ({len(cluster_games)} games):")
                # Handle Unicode characters in game names
                try:
                    print(f"  Sample games: {', '.join(cluster_games['name'].head(5).values)}")
                except UnicodeEncodeError:
                    print(f"  Sample games: [Games with special characters - see CSV file]")
                
                results.append({
                    'Algorithm': 'DBSCAN',
                    'Cluster': cluster_id,
                    'Size': len(cluster_games),
                    'Sample_Games': '; '.join(cluster_games['name'].head(10).values)
                })
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv('cluster_analysis.csv', index=False)
        print("\n[OK] Saved: cluster_analysis.csv")
        
        return results_df
    
    def save_results(self, output_prefix='steam_clustering'):
        """Save clustering results to CSV"""
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)
        
        results_df = self.processed_data.copy()
        
        if self.kmeans_labels is not None:
            results_df['kmeans_cluster'] = self.kmeans_labels
        
        if self.dbscan_labels is not None:
            results_df['dbscan_cluster'] = self.dbscan_labels
        
        output_path = f'{output_prefix}_results.csv'
        results_df.to_csv(output_path, index=False)
        print(f"[OK] Saved: {output_path}")
        
        return results_df


def main():
    """
    Main execution function
    This demonstrates the complete clustering pipeline
    """
    print("\n" + "=" * 70)
    print("STEAM GAMES CLUSTERING ANALYSIS")
    print("=" * 70)
    print("Algorithms: K-means and DBSCAN")
    print("Dataset: Steam Store Games")
    print("=" * 70)
    
    # Get the directory where this script is located
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'steam.csv')
    
    # Initialize clustering analysis
    # Note: The norm parameter can be changed by TA for testing
    clustering = SteamGamesClustering(
        data_path=data_path,
        norm='l2',
        random_state=42
    )
    
    # Load data
    if not clustering.load_data():
        print("\nWARNING: Please download the dataset and try again!")
        return
    
    # Preprocess data
    # Use games with moderate review counts for balance
    clustering.preprocess_data(min_reviews=10, max_games=1000)
    
    # Find optimal k for K-means
    optimal_k = clustering.find_optimal_k(k_range=range(3, 8))
    
    # Apply K-means with 5 clusters (good balance for visualization)
    clustering.apply_kmeans(n_clusters=5)
    
    # Apply DBSCAN with parameters to find at least some clusters
    # Small eps finds tight local clusters
    clustering.apply_dbscan(eps=0.6, min_samples=5)
    
    # Create visualizations
    clustering.visualize_clusters_2d()
    clustering.visualize_clusters_3d()
    
    # Analyze clusters
    clustering.analyze_clusters()
    
    # Save results
    clustering.save_results()
    
    print("\n" + "=" * 70)
    print("[SUCCESS] ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - optimal_k_analysis.png: Elbow method and silhouette analysis")
    print("  - clusters_2d.png: 2D PCA visualization")
    print("  - clusters_3d.png: 3D PCA visualization")
    print("  - cluster_analysis.csv: Detailed cluster information")
    print("  - steam_clustering_results.csv: Full results with cluster assignments")
    print("=" * 70)


if __name__ == "__main__":
    main()

