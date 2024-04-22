import folium
import pandas as pd


def load_data_file(abundance_file, metadata_file) :
    abundance_df = pd.read_csv(abundance_file, index_col=0)
    meta_df = pd.read_csv(metadata_file)
    return abundance_df, meta_df

def import_coordinates(abundance_df, meta_df) :
    coordinates = meta_df[['uuid', 'longitude', 'latitude']].copy()  # Copying the slice to a new DataFrame
    coordinates['uuid'] = coordinates['uuid'].astype(str)    
    abundance_df['uuid'] = abundance_df['uuid'].astype(str)
    coordinates['uuid'] = coordinates['uuid'].astype(str)
    df = pd.merge(abundance_df, coordinates, on='uuid', how="inner")
    df = df.dropna()
    return df


validation_data, metadata = load_data_file("validation_200.csv", "complete_metadata.csv")
validation_data = import_coordinates(validation_data, metadata)

latitudes = validation_data["latitude"].tolist()
longitudes = validation_data["longitude"].tolist()
coordinates = list(zip(latitudes, longitudes))

if coordinates:
    map = folium.Map(location=[coordinates[0][0], coordinates[0][1]], zoom_start=5)

    # Add markers for each coordinate in the list
    for coord in coordinates:
        folium.Marker(location=[coord[0], coord[1]], popup=f"Lat: {coord[0]}, Lon: {coord[1]}").add_to(map)

    # Display the map
    map.save("map.html")