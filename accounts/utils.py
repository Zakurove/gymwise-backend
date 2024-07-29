from django.conf import settings

def get_tenant_domain(request):
    host = request.get_host().lower()
    split_host = host.split('.')
    
    main_domain = settings.MAIN_DOMAIN
    domain = None
    subdomain = None

    if main_domain in host:
        if host == main_domain:
            return None, None
        subdomain = host.replace(f".{main_domain}", "")
    else:
        domain = host

    return domain, subdomain