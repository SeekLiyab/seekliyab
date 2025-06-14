import streamlit as st
from twilio.rest import Client
from datetime import datetime, timedelta
from services.database import get_supabase_connection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Twilio client
try:
    client = Client(st.secrets["twilio"]["account_sid"], st.secrets["twilio"]["auth_token"])
except Exception as e:
    logger.error(f"Failed to initialize Twilio client: {e}")
    client = None

def check_for_sms(df, column_name='fire_risk', check_records=20, min_consecutive=8):
    """
    Ultra-optimized version - checks for consecutive SAME fire type alerts
    
    Args:
        df: DataFrame with fire detection predictions  
        column_name: Column name containing predictions
        check_records: Number of records to check
        min_consecutive: Minimum consecutive alerts needed (same type)
    
    Returns:
        bool: True if SMS alert should be sent
    """
    # Fast validation
    if df.empty or len(df) < min_consecutive:
        return False
    
    # Get values as numpy array for speed
    values = df[column_name].head(check_records).values
    
    # Count consecutive occurrences of the SAME fire type from start
    if len(values) == 0:
        return False
    
    first_value = values[0]
    
    # Only count if first value is a fire condition
    if first_value not in ['Fire', 'Potential Fire']:
        return False
    
    # Count consecutive occurrences of the SAME fire type
    consecutive_count = 0
    for value in values:
        if value == first_value:  # Must be exactly the same type
            consecutive_count += 1
            if consecutive_count >= min_consecutive:
                return True
        else:
            break  # Stop at first different value
    
    return False


def log_notification_to_database(content, classification, sent_to_numbers):
    """
    Log notification details to the Supabase notifications table.
    
    Args:
        content (str): The SMS message content
        classification (int): Classification code (0=Potential Fire, 1=Fire)
        sent_to_numbers (list): List of phone numbers the SMS was sent to
    
    Returns:
        bool: True if logged successfully, False otherwise
    """
    try:
        conn = get_supabase_connection()
        
        # Prepare notification data
        notification_data = {
            "content": content,
            "classification": classification,
            "sent_to": ", ".join(sent_to_numbers),
            "device_alarm_triggered_at": datetime.now().isoformat()
        }
        
        # Insert into notifications table
        result = conn.table("notifications").insert(notification_data).execute()
        
        if result.data:
            logger.info(f"Notification logged to database: {content[:50]}...")
            return True
        else:
            logger.error("Failed to log notification to database")
            return False
            
    except Exception as e:
        logger.error(f"Error logging notification to database: {e}")
        return False


def check_recent_sms_sent(area_name, fire_risk, cooldown_minutes=15):
    """
    Check if an SMS has already been sent recently for the same area and fire risk.
    
    Args:
        area_name (str): The area name
        fire_risk (str): The fire risk level ('Fire' or 'Potential Fire')
        cooldown_minutes (int): Cooldown period in minutes (default: 15)
    
    Returns:
        bool: True if SMS was sent recently (within cooldown), False otherwise
    """
    try:
        conn = get_supabase_connection()
        
        # Calculate the cutoff time
        cutoff_time = datetime.now() - timedelta(minutes=cooldown_minutes)
        cutoff_time_iso = cutoff_time.isoformat()
        
        # Create search content pattern
        search_pattern = f"%{fire_risk.upper()}%{area_name}%"
        
        # Query recent notifications
        result = conn.table("notifications") \
            .select("id, content, created_at") \
            .ilike("content", search_pattern) \
            .gte("created_at", cutoff_time_iso) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
        
        if result.data and len(result.data) > 0:
            last_sms_time = datetime.fromisoformat(result.data[0]['created_at'].replace('Z', '+00:00'))
            time_diff = datetime.now() - last_sms_time.replace(tzinfo=None)
            logger.info(f"Last SMS for {fire_risk} in {area_name} was {time_diff.total_seconds()/60:.1f} minutes ago")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking recent SMS: {e}")
        # If we can't check, allow sending to be safe
        return False


def check_if_sent(area_name, fire_risk, lookback_hours=1):
    """
    Check if an SMS has already been sent for the same area and classification 
    within the specified time period.
    
    Args:
        area_name (str): The area name
        fire_risk (str): The fire risk level ('Fire' or 'Potential Fire')  
        lookback_hours (int): Hours to look back (default: 1)
    
    Returns:
        bool: True if SMS was already sent within the time period, False otherwise
    """
    try:
        conn = get_supabase_connection()
        
        # Calculate the cutoff time (1 hour ago by default)
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        cutoff_time_iso = cutoff_time.isoformat()
        
        # Map fire_risk to classification code
        classification_code = 1 if fire_risk == "Fire" else 0
        
        # Create search pattern for the area name in content
        area_search_pattern = f"%{area_name}%"
        
        # Query recent notifications with same area and classification
        result = conn.table("notifications") \
            .select("id, content, classification, created_at") \
            .eq("classification", classification_code) \
            .ilike("content", area_search_pattern) \
            .gte("created_at", cutoff_time_iso) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
        
        if result.data and len(result.data) > 0:
            last_notification = result.data[0]
            last_sms_time = datetime.fromisoformat(last_notification['created_at'].replace('Z', '+00:00'))
            time_diff = datetime.now() - last_sms_time.replace(tzinfo=None)
            
            logger.info(f"Found recent SMS for {fire_risk} in {area_name}: {time_diff.total_seconds()/3600:.1f} hours ago")
            return True
        
        logger.info(f"No recent SMS found for {fire_risk} in {area_name} within {lookback_hours} hour(s)")
        return False
        
    except Exception as e:
        logger.error(f"Error checking if SMS was sent: {e}")
        # If we can't check, return False to allow sending (fail-safe)
        return False


def send_sms(area_name, fire_risk):
    """
    Send SMS alerts to emergency contacts and log to database.
    Includes anti-spam protection to prevent duplicate messages.
    
    Args:
        area_name (str): The area where fire risk is detected
        fire_risk (str): The fire risk level ('Fire' or 'Potential Fire')
    
    Returns:
        dict: Contains 'sent' (bool), 'reason' (str), 'blocked_by_cooldown' (bool)
    """
    if not client:
        logger.error("Twilio client not initialized")
        return {"sent": False, "reason": "Twilio client not initialized", "blocked_by_cooldown": False}
    
    # Check if we've sent SMS recently for this area/risk combination
    if check_recent_sms_sent(area_name, fire_risk):
        logger.info(f"SMS blocked by cooldown for {fire_risk} in {area_name}")
        return {
            "sent": False, 
            "reason": f"SMS already sent recently for {fire_risk} in {area_name}. Cooldown active to prevent spam.",
            "blocked_by_cooldown": True
        }
    
    try:
        # Get emergency numbers from secrets
        emergency_numbers = st.secrets["emergency_contacts"]["emergency_numbers"]
        
        # Handle both single string and list of numbers
        if isinstance(emergency_numbers, str):
            phone_numbers = [emergency_numbers]
        else:
            phone_numbers = emergency_numbers
        
        # Create appropriate message based on fire risk level
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S PHT")
        
        if fire_risk == "Fire":
            sms_message = f"🚨 SEEKLIYAB EMERGENCY ALERT 🚨\n\nFIRE DETECTED in {area_name} at {current_time}!\n\nIMMEDIATE EVACUATION REQUIRED!\n\nPlease respond immediately."
            classification = 1  # Fire classification
        else:  # Potential Fire
            sms_message = f"⚠️ SEEKLIYAB FIRE ALERT ⚠️\n\nPOTENTIAL FIRE detected in {area_name} at {current_time}!\n\nPlease investigate the area immediately.\n\nStay alert and prepared."
            classification = 0  # Potential Fire classification
        
        # Send SMS to all emergency contacts
        successful_sends = []
        failed_sends = []
        
        for phone_number in phone_numbers:
            try:
                message = client.messages.create(
                    to=phone_number,
                    from_=st.secrets["twilio"]["from_number"],
                    body=sms_message
                )
                successful_sends.append(phone_number)
                logger.info(f"SMS sent successfully to {phone_number}: {message.sid}")
                
            except Exception as e:
                failed_sends.append(phone_number)
                logger.error(f"Failed to send SMS to {phone_number}: {e}")
        
        # Log notification to database if at least one SMS was sent successfully
        if successful_sends:
            log_notification_to_database(sms_message, classification, successful_sends)
            logger.info(f"SMS alert sent for {fire_risk} in {area_name} to {len(successful_sends)} recipients")
            return {
                "sent": True, 
                "reason": f"SMS sent successfully to {len(successful_sends)} recipients",
                "blocked_by_cooldown": False
            }
        else:
            logger.error(f"Failed to send SMS to any recipients for {fire_risk} in {area_name}")
            return {
                "sent": False, 
                "reason": f"Failed to send SMS to any recipients",
                "blocked_by_cooldown": False
            }
            
    except Exception as e:
        logger.error(f"Error in send_sms function: {e}")
        return {
            "sent": False, 
            "reason": f"Error in SMS service: {str(e)}",
            "blocked_by_cooldown": False
        }