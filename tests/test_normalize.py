import unittest

from src.normalize import normalize_mention


HONORIFICS = ["Lt. Gen.", "Dr.", "Mr."]
SUFFIXES = ["Pvt. Ltd.", "Limited", "Ltd."]


class NormalizeTests(unittest.TestCase):
    def test_person_honorific_unicode_and_initials(self):
        full = normalize_mention("\u200bDr.  Rajesh Sharma", "PERSON", HONORIFICS, SUFFIXES)
        abbreviated = normalize_mention("R. Sharma", "PERSON", HONORIFICS, SUFFIXES)
        self.assertEqual(full.key, "rajesh sharma")
        self.assertEqual(full.initials_key, abbreviated.initials_key)
        self.assertEqual(full.display, "\u200bDr.  Rajesh Sharma")

    def test_organization_suffix_and_acronym(self):
        full = normalize_mention("Acme Pvt. Ltd.", "ORG", HONORIFICS, SUFFIXES)
        barc = normalize_mention("Bhabha Atomic Research Centre", "ORG", HONORIFICS, SUFFIXES)
        self.assertEqual(full.key, "acme")
        self.assertEqual(barc.initials_key, "barc")


if __name__ == "__main__":
    unittest.main()

