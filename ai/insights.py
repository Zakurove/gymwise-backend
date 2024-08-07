from datetime import timedelta
from django.utils import timezone
from hijri_converter import convert
from django.db.models import Count
import logging

logger = logging.getLogger(__name__)

def generate_insights(member, historical_data):
    insights = []
    now = timezone.now()
    
    logger.info(f"Generating insights for member {member.id}")
    
    if member.churn_risk == 'high':
        insights.append({
            'type': 'high_risk',
            'message': f"High risk of churn detected for {member.name}."
        })
    elif member.churn_risk == 'medium':
        insights.append({
            'type': 'medium_risk',
            'message': "Moderate churn risk detected."
        })


    if member.visit_frequency is not None:
        if member.visit_frequency < 2:
            insights.append({
                'type': 'low_engagement',
                'message': "Low visit frequency detected."
            })
        elif member.visit_frequency > 10:
            insights.append({
                'type': 'high_engagement',
                'message': "Highly engaged member identified."
            })

    if member.membership_duration is not None and member.churn_risk != 'low':
        if member.membership_duration > 12:
            insights.append({
                'type': 'at_risk_veteran',
                'message': "Long-time member showing signs of disengagement."
            })

    if historical_data.exists():
        recent_visits = historical_data.filter(date__gte=now.date() - timedelta(days=30)).count()
        avg_monthly_visits = member.visit_frequency * 4 if member.visit_frequency is not None else 0
        if recent_visits < avg_monthly_visits * 0.7:
            insights.append({
                'type': 'decreasing_activity',
                'message': "Recent activity has decreased compared to usual patterns."
            })

    current_hijri = convert.Gregorian(now.year, now.month, now.day).to_hijri()
    
    ramadan_start = convert.Hijri(current_hijri.year, 9, 1).to_gregorian()
    ramadan_end = convert.Hijri(current_hijri.year, 10, 1).to_gregorian()
    if ramadan_start <= now.date() <= ramadan_end:
        ramadan_visits = historical_data.filter(date__range=[ramadan_start, ramadan_end]).count()
        if member.visit_frequency is not None and ramadan_visits < member.visit_frequency * 2:
            insights.append({
                'type': 'ramadan_decrease',
                'message': "Member's attendance during Ramadan is lower than usual."
            })
    elif (ramadan_start - now.date()).days <= 30:
        insights.append({
            'type': 'pre_ramadan',
            'message': "Ramadan is approaching. Attendance patterns may change."
        })

    eid_al_fitr = convert.Hijri(current_hijri.year, 10, 1).to_gregorian()
    eid_al_adha = convert.Hijri(current_hijri.year, 12, 10).to_gregorian()
    if (eid_al_fitr - now.date()).days <= 7 or (eid_al_adha - now.date()).days <= 7:
        insights.append({
            'type': 'eid_approaching',
            'message': "Eid is approaching. Expect potential changes in attendance patterns."
        })

    if 5 <= now.month <= 8:  # May to August
        summer_visits = historical_data.filter(date__month__in=[5, 6, 7, 8]).count()
        if member.visit_frequency is not None and summer_visits < member.visit_frequency * 8:
            insights.append({
                'type': 'summer_decrease',
                'message': "Member's summer attendance is lower than usual."
            })

    favorite_class = historical_data.values('class_name').annotate(count=Count('id')).order_by('-count').first()
    if favorite_class:
        insights.append({
            'type': 'class_preference',
            'message': f"Member shows a strong preference for {favorite_class['class_name']} classes."
        })

    preferred_time = historical_data.values('time_of_day').annotate(count=Count('id')).order_by('-count').first()
    if preferred_time:
        insights.append({
            'type': 'time_preference',
            'message': f"Member often visits during {preferred_time['time_of_day']} hours."
        })


    logger.info(f"Generated {len(insights)} insights for member {member.id}")

    return insights