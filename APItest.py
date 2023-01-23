import unittest
from MainPage import API, load_image
import requests_mock

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.api_key = "your_api_key"
        self.api = API(self.api_key)
    
    @requests_mock.Mocker()
    def test_autocomplete(self, mock):
        # Test data
        parameters = "Fort McMurray"
        test_data = {'query': {'parsed': {'city': 'fort mcmurray', 'expected_type': 'unknown'},
           'text': 'Fort McMurray'},
 'results': [{'address_line1': 'Fort McMurray, AB',
              'address_line2': 'Canada',
              'bbox': {'lat1': 56.6437458,
                       'lat2': 56.8044658,
                       'lon1': -111.5109382,
                       'lon2': -111.2070503},
              'category': 'administrative',
              'city': 'Fort McMurray',
              'country': 'Canada',
              'country_code': 'ca',
              'county': 'Wood Buffalo',
              'datasource': {'attribution': '© OpenStreetMap contributors',
                             'license': 'Open Database License',
                             'sourcename': 'openstreetmap',
                             'url': 'https://www.openstreetmap.org/copyright'},
              'formatted': 'Fort McMurray, AB, Canada',
              'lat': 56.7291997,
              'lon': -111.3885221,
              'place_id': '517d51cc8bddd85bc05963e06f6a565d4c40f00101f9017231730000000000c00208',
              'rank': {'confidence': 1,
                       'confidence_city_level': 1,
                       'importance': 0.6899868129468802,
                       'match_type': 'full_match',
                       'popularity': 0.7940101998988992},
              'result_type': 'city',
              'state': 'Alberta',
              'state_code': 'AB',
              'timezone': {'abbreviation_DST': 'MDT',
                           'abbreviation_STD': 'MST',
                           'name': 'America/Edmonton',
                           'offset_DST': '-06:00',
                           'offset_DST_seconds': -21600,
                           'offset_STD': '-07:00',
                           'offset_STD_seconds': -25200}},
             {'address_line1': 'Vantage Inns & Suites, Fort McMurray',
              'address_line2': '200 Parent Way, Fort McMurray, AB T9H 5E6, '
                               'Canada',
              'bbox': {'lat1': 56.6838519,
                       'lat2': 56.6843986,
                       'lon1': -111.3508741,
                       'lon2': -111.3505123},
              'category': 'accommodation.hotel',
              'city': 'Fort McMurray',
              'country': 'Canada',
              'country_code': 'ca',
              'county': 'Wood Buffalo',
              'datasource': {'attribution': '© OpenStreetMap contributors',
                             'license': 'Open Database License',
                             'sourcename': 'openstreetmap',
                             'url': 'https://www.openstreetmap.org/copyright'},
              'formatted': 'Vantage Inns & Suites, Fort McMurray, 200 Parent '
                           'Way, Fort McMurray, AB T9H 5E6, Canada',
              'housenumber': '200',
              'lat': 56.68412525,
              'lon': -111.3506932,
              'name': 'Vantage Inns & Suites, Fort McMurray',
              'place_id': '517e3be4c171d65bc059158f8b6a91574c40f00102f9013cded61800000000c0020192032456616e7461676520496e6e732026205375697465732c20466f7274204d634d7572726179',
              'postcode': 'T9H 5E6',
              'rank': {'confidence': 1,
                       'importance': 0.20000999999999997,
                       'match_type': 'full_match',
                       'popularity': 0.4889311654178562},
              'result_type': 'amenity',
              'state': 'Alberta',
              'state_code': 'AB',
              'street': 'Parent Way',
              'suburb': 'Gregoire',
              'timezone': {'abbreviation_DST': 'MDT',
                           'abbreviation_STD': 'MST',
                           'name': 'America/Edmonton',
                           'offset_DST': '-06:00',
                           'offset_DST_seconds': -21600,
                           'offset_STD': '-07:00',
                           'offset_STD_seconds': -25200}}]}
        
        mock.get(f"https://api.geoapify.com/v1/geocode/search?text={parameters}&format=json&apiKey={self.api_key}", json=test_data)
        
        # Test autocomplete
        data = self.api.autocomplete(parameters)
        self.assertEqual(data, test_data)
        # print success message
        print("Test autocomplete passed")
        
    def test_location_to_API(self):
        location = "Fort McMurray"
        expected_location = "Fort%20McMurray"
        self.assertEqual(self.api.location_to_API(location), expected_location)
        # print success message
        print("Test location_to_API passed")
if __name__ == '__main__':
    unittest.main()