def allowed_domains_for_query(query):
    return ["example.com"]

def select_sources_for_query(query, domains=None):
    return domains or ["example.com"]
