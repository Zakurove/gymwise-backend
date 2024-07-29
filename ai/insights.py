from datetime import timedelta
from django.utils import timezone
from hijri_converter import Gregorian, Hijri
from django.db.models import Count

def generate_insights(member, historical_data):
    insights = []
    now = timezone.now()
    
    # Churn Risk Insights
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

    # Visit Frequency Insights
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

    # Membership Duration Insights
    if member.membership_duration > 12 and member.churn_risk != 'low':
        insights.append({
            'type': 'at_risk_veteran',
            'message': "Long-time member showing signs of disengagement."
        })

    # Recent Activity Trend
    recent_visits = historical_data.filter(date__gte=now - timedelta(days=30)).count()
    avg_monthly_visits = member.visit_frequency * 4  # Assuming visit_frequency is per week
    if recent_visits < avg_monthly_visits * 0.7:
        insights.append({
            'type': 'decreasing_activity',
            'message': "Recent activity has decreased compared to usual patterns."
        })

    # Seasonal Trend (Saudi Arabia specific)
    current_hijri = Hijri.from_gregorian(now.year, now.month, now.day)
    
    # Ramadan insights
    ramadan_start = Hijri(current_hijri.year, 9, 1).to_gregorian()
    ramadan_end = Hijri(current_hijri.year, 10, 1).to_gregorian()
    if ramadan_start <= now.date() <= ramadan_end:
        ramadan_visits = historical_data.filter(date__range=[ramadan_start, ramadan_end]).count()
        if ramadan_visits < member.visit_frequency * 2:  # Assuming at least 2 weeks of Ramadan
            insights.append({
                'type': 'ramadan_decrease',
                'message': "Member's attendance during Ramadan is lower than usual."
            })
    elif (ramadan_start - now.date()).days <= 30:
        insights.append({
            'type': 'pre_ramadan',
            'message': "Ramadan is approaching. Attendance patterns may change."
        })

    # Eid insights
    eid_al_fitr = Hijri(current_hijri.year, 10, 1).to_gregorian()
    eid_al_adha = Hijri(current_hijri.year, 12, 10).to_gregorian()
    if (eid_al_fitr - now.date()).days <= 7 or (eid_al_adha - now.date()).days <= 7:
        insights.append({
            'type': 'eid_approaching',
            'message': "Eid is approaching. Expect potential changes in attendance patterns."
        })

    # Summer insights
    if 5 <= now.month <= 8:  # May to August
        summer_visits = historical_data.filter(date__month__in=[5, 6, 7, 8]).count()
        if summer_visits < member.visit_frequency * 8:  # Assuming at least 8 weeks of summer
            insights.append({
                'type': 'summer_decrease',
                'message': "Member's summer attendance is lower than usual."
            })

    # Class Preference
    favorite_class = historical_data.values('class_name').annotate(count=Count('id')).order_by('-count').first()
    if favorite_class:
        insights.append({
            'type': 'class_preference',
            'message': f"Member shows a strong preference for {favorite_class['class_name']} classes."
        })

    # Time Preference
    preferred_time = historical_data.values('time_of_day').annotate(count=Count('id')).order_by('-count').first()
    if preferred_time:
        insights.append({
            'type': 'time_preference',
            'message': f"Member often visits during {preferred_time['time_of_day']} hours."
        })

    return insights