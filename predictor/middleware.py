from .models import AuditTrail
from django.utils.deprecation import MiddlewareMixin

class AuditTrailMiddleware(MiddlewareMixin):
    def process_view(self, request, view_func, view_args, view_kwargs):
        if request.user.is_authenticated:
            path = request.path
            method = request.method
            action = f"{method} {path}"

            # Avoid logging static/media/admin
            if not path.startswith('/static') and 'admin' not in path:
                AuditTrail.objects.create(
                    user=request.user,
                    action=action,
                    details=f"User accessed {view_func.__name__} view"
                )