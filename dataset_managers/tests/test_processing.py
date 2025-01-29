import unittest
import preprocessing_utils

class ProcessingTest(unittest.TestCase):
    def test_replace_preds_v1(self):

        relations_dict = {"P740": {"label": "location of formation",
                                   "description": "location where a group or organization was formed"},
                          "P17": {"label": "country",
                                  "description": "sovereign state of this item; don't use on humans"}}

        sparql = "SELECT ?answer WHERE { wd:Q748 wdt:P740 [ wdt:P17 ?answer ] .}"
        sparql_with_predicates = preprocessing_utils.replace_preds(sparql, relations_dict)
        expected_sparql = "SELECT ?answer WHERE { wd:Q748 location_of_formation [ country ?answer ] .}"

        self.assertEqual(sparql_with_predicates, expected_sparql)

    def test_replace_preds_v2(self):

        relations_dict = {"P740": {"label": "location of formation",
                                   "description": "location where a group or organization was formed"},
                          "P17": {"label": "country",
                                  "description": "sovereign state of this item; don't use on humans"}}

        sparql = "SELECT ?answer WHERE { wd:Q38104 p:P1346 [ps:P1346 ?answer; pq:P585 ?year] } ORDER BY ASC(?year) LIMIT 1"
        sparql_with_predicates = preprocessing_utils.replace_preds(sparql, relations_dict)
        expected_sparql = "SELECT ?answer WHERE { wd:Q38104 p:P1346 [ps:P1346 ?answer; pq:P585 ?year] } ORDER BY ASC(?year) LIMIT 1"

        self.assertEqual(sparql_with_predicates, expected_sparql)