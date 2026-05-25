from mpn import extract_mpn_from_url


def test_extract_mpn_from_query_param() -> None:
    url = "https://shop.example.com/laptop?sku=21ABC123"
    assert extract_mpn_from_url(url) == "21ABC123"


def test_extract_mpn_from_path_segment() -> None:
    url = "https://www.lenovo.com/us/en/p/laptops/thinkpad/21N3001BUS"
    assert extract_mpn_from_url(url) == "21N3001BUS"
