# ElasticON 2025 - Forge the Future Hackathon
This repo is created to host the code and development for "Forge the Future Hackathon"

The goal of this repo is to share how team Helios combines data from **Singapore's OpenData (thanks GovTech)** to:
- Enable anyone in Singapore ask questions on where the closest park is to them given a road name or postal code
- Allows one to plan a Community Health Assistance Scheme (CHAS) clinic that may also coincide with other convenient Social Care facilities
- For foodies and health enthusiasts to find their hawker place at the same locating the closest parks or gyms for exercises

Some use cases that we hope to unlock for our users
1. Ability to converse with the OpenData sets that are published and made available 
2. Ability to find the necessary support or help depending on their circumstances
3. Using Kibana maps to visualise probable proximity locations and facilities

Thanks to these tools and sponsors:
1. Elastic - for sponsoring the Elasticsearch Cluster in the Cloud
2. AWS - for sponsoring Amazon Bedrock that is being used for our AI Chatbots
3. Tines - for an amazingly easy to use automation tool to ingest all the necessary data from [OpenData](https://data.gov.sg)
4. GovTech & Singapore Agencies - for diligently publishing data to their portal ordinary users' consumption and analysis

## Setup Instructions

1. Install required Python packages:
```
pip install -r requirements.txt
```

2. Create a `.env` file in the app directory with the following variables:
```
ES_API_KEY=your_es_api_key
AWS_ACCESS_KEY=your_aws_access_key
AWS_SECRET_KEY=your_aws_secret_key
```

3. Run the application:
```
python hackathon_final.py
```
