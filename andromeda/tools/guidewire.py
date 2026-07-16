from __future__ import annotations
 
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import base64
import io
from typing import Any, Dict, Optional, Union
from functools import lru_cache
from pypdf import PdfReader
from langchain_core.tools import tool
import json
import tiktoken
 
 
 
 
class GuidewireClient:
    def __init__(self, base_url: str, username: str, password: str, *, timeout_seconds: int = 20) -> None:
            base_url = (base_url or "").strip()
            if not base_url:
                raise ValueError("Guidewire base_url is required.")
            self.base_url = base_url.rstrip("/")
            self.timeout_seconds = timeout_seconds
            self.username = username
            self.password = password
 
    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"
 
    def _get_auth(self) -> str:
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode("utf-8")).decode("ascii")
        return f"Basic {encoded}"
    
    @lru_cache(maxsize=256)
    def get_document_text_content(self, claim_id: str, document_id: str) -> Dict[str, Any]:
        safe_claim = urllib.parse.quote(str(claim_id), safe="%")
        safe_document_id = urllib.parse.quote(str(document_id), safe="%")
 
        content_response = self.get_json(
            f"rest/claim/v1/claims/{safe_claim}/documents/{safe_document_id}/content"
        )
 
        extracted_text = None
        fetch_status = "no_content"
 
        if isinstance(content_response, dict) and not content_response.get("error"):
            pdf_base64 = (
                content_response.get("data", {})
                .get("attributes", {})
                .get("contents")
            )
 
            if pdf_base64:
                extracted_text = self.pdf_base64_to_text(pdf_base64)
                fetch_status = "success" if extracted_text else "text_extraction_failed"
            else:
                fetch_status = "contents_missing"
 
        elif isinstance(content_response, dict) and content_response.get("error"):
            fetch_status = "content_api_failed"
 
        return {
            "document_text": extracted_text,
            "fetch_status": fetch_status,
            "content_error": content_response if isinstance(content_response, dict) and content_response.get("error") else None,
        }
    @lru_cache(maxsize=256)
    def pdf_base64_to_text(self, pdf_base64: str) -> Optional[str]:
        try:
            pdf_bytes = base64.b64decode(pdf_base64)
            pdf_stream = io.BytesIO(pdf_bytes)
            reader = PdfReader(pdf_stream)
 
            extracted_pages = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                extracted_pages.append(page_text)
 
            full_text = "\n".join(extracted_pages).strip()
            return full_text if full_text else None
 
        except Exception as exc:
            print(f"PDF text extraction failed: {exc}")
            return None
    @lru_cache(maxsize=256)
    def grep_file(self,pattern: str, text: str) -> str:
        """
        Search for a pattern and return matching lines from the provided PDF text content.
 
        Args:
            pattern: The pattern to search for in the text
            text: Extracted PDF text content to search in
        Returns:
            str: The first 20 matching lines with 1 line above and 1 line below the matching line if found.
        """
        if text is None or str(text).strip() == "":
            return "Provided text is empty."
 
        try:
            lines = str(text).splitlines()
 
            matching_lines = [
                (f"Line {i}: {lines[i-1]}\n" if i > 0 else "") +
                f"Line {i+1}: {line}\n" +
                (f"Line {i+2}: {lines[i+1]}\n" if i+1 < len(lines) else "")
                for i, line in enumerate(lines)
                if re.search(pattern, line, re.IGNORECASE)
            ]
 
            if len(matching_lines) > 20:
                matching_lines = matching_lines[:20] + [f"... {len(matching_lines) - 20} more lines truncated..."]
 
            if matching_lines:
                return "\n".join(matching_lines)
            else:
                return f"No matches found for pattern '{pattern}' in provided text."
 
        except re.error as e:
            return f"Invalid regex pattern: {str(e)}"
        except Exception as e:
            return f"Error searching text: {str(e)}"
    
    def truncate_to_token_limit(self,data: list[dict], max_tokens: int = 4000, model: str = "gpt-4o-mini") -> tuple[str, bool]:
        results_str = json.dumps(data, ensure_ascii=False, indent=2)
 
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
 
        tokens = enc.encode(results_str)
 
        if len(tokens) <= max_tokens:
            return results_str, False
 
        truncated_str = enc.decode(tokens[:max_tokens])
        return truncated_str + "\n... truncated due to token limit ...", True
 
    def get_json(self, path: str, *, headers: Optional[Dict[str, str]] = None) -> Union[Dict[str, Any], str]:
        url = self._url(path)
        req = urllib.request.Request(url, method="GET")
 
        request_headers = dict(headers or {})
        request_headers["Authorization"] = self._get_auth()
        request_headers["Accept"] = "application/json"
 
        for key, value in request_headers.items():
            req.add_header(key, value)
 
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:  # noqa: S310
                content_type = (resp.headers.get("Content-Type") or "").lower()
                raw = resp.read()
 
                if not raw:
                    return {"url": url, "status": resp.status, "data": None}
 
                if "application/json" in content_type:
                    return json.loads(raw.decode("utf-8"))
 
                return raw.decode("utf-8", errors="replace")
 
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
 
            return {
                "error": "http_error",
                "url": url,
                "status": exc.code,
                "reason": str(getattr(exc, "reason", "")) or None,
                "body": body or None,
            }
 
        except urllib.error.URLError as exc:
            return {"error": "url_error", "url": url, "reason": str(exc.reason)}
 
        except Exception as exc:  # noqa: BLE001
            return {"error": "unexpected_error", "url": url, "detail": str(exc)}
 
 
def make_guidewire_tools(base_url: str, username: str, password: str) -> Dict[str, Any]:
    gw = GuidewireClient(base_url, username, password)
 
    @tool
    def get_claim_details(claim_id: str) -> Union[Dict[str, Any], str]:
        """
        Fetch claim details from a Guidewire API.
 
        Args:
            claim_id: Claim identifier (e.g., claim ID).
 
        Returns:
            Parsed JSON response (dict) when the API returns JSON; otherwise a string.
        """
        safe_claim = urllib.parse.quote(str(claim_id), safe="%")
        raw = gw.get_json(f"rest/claim/v1/claims/{safe_claim}")
        data = raw.get("data", {}) if isinstance(raw, dict) else {}
        attrs = data.get("attributes", {}) if isinstance(data, dict) else {}
 
        def nested_name(value):
            if isinstance(value, dict):
                return value.get("displayName") or value.get("name") or value.get("code")
            return value
 
        policy_addresses = attrs.get("policyAddresses")
        if isinstance(policy_addresses, list):
            policy_address = ", ".join(
                addr.get("displayName")
                for addr in policy_addresses
                if isinstance(addr, dict) and addr.get("displayName")
            ) or None
        elif isinstance(policy_addresses, dict):
            policy_address = policy_addresses.get("displayName")
        else:
            policy_address = policy_addresses
 
        return {
            "claim_number": attrs.get("claimNumber") or attrs.get("claim_number"),
            "policy_number": attrs.get("policyNumber") or attrs.get("policy_number"),
            "status": nested_name(attrs.get("state")) or attrs.get("claim_status"),
            "loss_type": nested_name(attrs.get("lossType")),
            "loss_cause": nested_name(attrs.get("lossCause")),
            "loss_date": attrs.get("lossDate"),
            "insured": nested_name(attrs.get("insured")),
            "reporter": nested_name(attrs.get("reporter")),
            "assigned_adjuster": nested_name(attrs.get("assignedUser")) or attrs.get("adjuster"),
            "jurisdiction": nested_name(attrs.get("jurisdiction")),
            "line_of_business": nested_name(attrs.get("lobCode")),
            "how_reported": nested_name(attrs.get("howReported")),
            "reported_date": attrs.get("reportedDate"),
            "main_contact": nested_name(attrs.get("mainContact")),
            "loss_location": nested_name(attrs.get("lossLocation")),
            "policy_address": policy_address,
        }
 
    @tool
    def get_policy_details(claim_id: str) -> Union[Dict[str, Any], str]:
        """
        Fetch policy details associated with a claim from a Guidewire API.
 
        Args:
            claim_id: Claim identifier (e.g., claim ID).
        Returns:
            Parsed JSON response (dict) when the API returns JSON; otherwise a string.
        """
        safe_claim = urllib.parse.quote(str(claim_id), safe="%")
        raw = gw.get_json(f"rest/claim/v1/claims/{safe_claim}/policy")
        data = raw.get("data", {}) if isinstance(raw, dict) else {}
        attrs = data.get("attributes", {}) if isinstance(data, dict) else {}
 
        # Helper to extract nested code/name (like for currency or policyType)
        def nested_value(v):
            if isinstance(v, dict):
                return v.get("name") or v.get("code")
            return v
 
        return {
            "policy_number": attrs.get("policyNumber") or attrs.get("policy_number"),
            "policy_type": nested_value(attrs.get("policyType")),
            "policy_effective_date": attrs.get("effectiveDate"),
            "policy_expiration_date": attrs.get("expirationDate"),
            "currency": nested_value(attrs.get("currency")),
            # "verified_policy": attrs.get("verifiedPolicy"),
            "insured": attrs.get("insured", {}).get("displayName") if isinstance(attrs.get("insured"), dict) else attrs.get("insured"),
        }
 
    @tool
    def get_claim_activities(claim_id: str) -> Union[Dict[str, Any], str]:
        """
        Fetch activities associated with a claim from a Guidewire API.
 
        Args:
            claim_id: Claim identifier (e.g., claim ID).
 
        Returns:
            Parsed JSON response (dict) when the API returns JSON; otherwise a string.
        """
        safe_claim = urllib.parse.quote(str(claim_id), safe="%")
        raw = gw.get_json(f"rest/claim/v1/claims/{safe_claim}/activities")
        if not isinstance(raw, dict) or "data" not in raw or not isinstance(raw["data"], list):
            return raw
        def clean_activity(a):
            attrs = a.get("attributes", {})
            return {
                "id": attrs.get("id") or a.get("id"),
                "subject": attrs.get("subject"),
                "activity_pattern": attrs.get("activityPattern"),
                "activity_type": (attrs.get("activityType") or {}).get("name"),
                "assigned_user": (attrs.get("assignedUser") or {}).get("displayName"),
                "assigned_group": (attrs.get("assignedGroup") or {}).get("displayName"),
                "status": (attrs.get("status") or {}).get("name"),
                "priority": (attrs.get("priority") or {}).get("name"),
                "due_date": attrs.get("dueDate"),
                "create_time": attrs.get("createTime"),
                "mandatory": attrs.get("mandatory"),
                "escalated": attrs.get("escalated"),
                "externally_owned": attrs.get("externallyOwned"),
                "related_to": (attrs.get("relatedTo") or {}).get("displayName"),
                "assigned_by_user": (attrs.get("assignedByUser") or {}).get("displayName"),
                "assignment_status": (attrs.get("assignmentStatus") or {}).get("name"),
                "importance": (attrs.get("importance") or {}).get("name"),
                "description": attrs.get("description"),
            }
        cleaned_activities = [clean_activity(act) for act in raw["data"]]
        return {
            "count": len(cleaned_activities),
            "activities": cleaned_activities
        }
 
    @tool
    def get_claim_notes(claim_id: str) -> Union[Dict[str, Any], str]:
        """
        Fetch notes associated with a claim from a Guidewire API.
 
        Args:
            claim_id: Claim identifier (e.g., claim ID).
 
        Returns:
            Parsed JSON response (dict) when the API returns JSON; otherwise a string.
        """
        safe_claim = urllib.parse.quote(str(claim_id), safe="%")
        raw = gw.get_json(f"rest/claim/v1/claims/{safe_claim}/notes")
        if not isinstance(raw, dict) or "data" not in raw or not isinstance(raw["data"], list):
            return raw
 
        def clean_note(note):
            attrs = note.get("attributes", {})
            return {
                "id": attrs.get("id") or note.get("id"),
                "author": (attrs.get("author") or {}).get("displayName"),
                "created_date": attrs.get("createdDate"),
                "updated_date": attrs.get("updateTime"),
                "body_summary": attrs.get("bodySummary"),
                "confidential": attrs.get("confidential"),
                "topic": (attrs.get("topic") or {}).get("name"),
                "related_to": (attrs.get("relatedTo") or {}).get("displayName"),
            }
        cleaned_notes = [clean_note(n) for n in raw.get("data", [])]
 
        return {
            "count": len(cleaned_notes),
            "notes": cleaned_notes
        }
 
    @tool
    def get_claim_history_events(claim_id: str) -> Union[Dict[str, Any], str]:
        """
        Fetch history events associated with a claim from a Guidewire API.
 
        Args:
            claim_id: Claim identifier (e.g., claim ID).
 
        Returns:
            Parsed JSON response (dict) when the API returns JSON; otherwise a string.
        """
        safe_claim = urllib.parse.quote(str(claim_id), safe="%")
        raw = gw.get_json(f"rest/claim/v1/claims/{safe_claim}/history-events")
        if not isinstance(raw, dict) or "data" not in raw:
            return raw
        def clean_event(evt):
            a = evt.get("attributes", {})
            return {
                "timestamp": a.get("eventTimestamp"),
                "type": a.get("historyType", {}).get("name"),
                "description": a.get("description"),
                "user": a.get("user", {}).get("displayName"),
            }
        events = [clean_event(e) for e in raw.get("data", [])]
        return {"count": len(events), "events": events}

    @tool
    def get_claim_exposures(claim_id: str) -> Union[Dict[str, Any], str]:
        """
        Fetch exposures associated with a claim from a Guidewire API.

        Args:
            claim_id: Claim identifier (e.g., claim ID).

        Returns:
            Parsed JSON response (dict) when the API returns JSON; otherwise a string.
        """
        safe_claim = urllib.parse.quote(str(claim_id), safe="%")
        raw = gw.get_json(f"rest/claim/v1/claims/{safe_claim}/exposures")
        if not isinstance(raw, dict) or "data" not in raw or not isinstance(raw["data"], list):
            return raw

        def clean_exposure(exposure):
            attrs = exposure.get("attributes", {})
            return {
                # "id": attrs.get("id") or exposure.get("id"),
                "claim_order": attrs.get("claimOrder"),
                "status": (attrs.get("state") or {}).get("name"),
                "exposure_type": (attrs.get("type") or {}).get("name"),
                "loss_party": (attrs.get("lossParty") or {}).get("name"),
                "claimant": (attrs.get("claimant") or {}).get("displayName"),
                "claimant_type": (attrs.get("claimantType") or {}).get("name"),
                "primary_coverage": (attrs.get("primaryCoverage") or {}).get("name"),
                "coverage_subtype": (attrs.get("coverageSubtype") or {}).get("name"),
                "jurisdiction": (attrs.get("jurisdiction") or {}).get("name"),
                "segment": (attrs.get("segment") or {}).get("name"),
                "assigned_user": (attrs.get("assignedUser") or {}).get("displayName"),
                "assigned_group": (attrs.get("assignedGroup") or {}).get("displayName"),
                "assigned_by_user": (attrs.get("assignedByUser") or {}).get("displayName"),
                "assignment_status": (attrs.get("assignmentStatus") or {}).get("name"),
                "contact_permitted": attrs.get("contactPermitted"),
                "create_time": attrs.get("createTime"),
                "created_via": (attrs.get("createdVia") or {}).get("name"),
                "strategy": (attrs.get("strategy") or {}).get("name"),
                "tier": (attrs.get("tier") or {}).get("name"),
                "vehicle_incident": (attrs.get("vehicleIncident") or {}).get("displayName"),
            }

        cleaned_exposures = [clean_exposure(e) for e in raw.get("data", [])]

        return {
            "count": len(cleaned_exposures),
            "exposures": cleaned_exposures
        }

    @tool
    def get_claim_contacts(claim_id: str) -> Union[Dict[str, Any], str]:
        """
        Fetch contacts associated with a claim from a Guidewire API.

        Args:
            claim_id: Claim identifier (e.g., claim ID).

        Returns:
            Parsed JSON response (dict) when the API returns JSON; otherwise a string.
        """
        safe_claim = urllib.parse.quote(str(claim_id), safe="%")
        raw = gw.get_json(f"rest/claim/v1/claims/{safe_claim}/contacts")
        if not isinstance(raw, dict) or "data" not in raw or not isinstance(raw["data"], list):
            return raw

        def clean_contact(contact):
            attrs = contact.get("attributes", {})
            roles = []
            for r in (attrs.get("roles") or []):
                role_info = r.get("role") or {}
                related = r.get("relatedTo") or {}
                roles.append({
                    "role": role_info.get("name"),
                    "active": r.get("active"),
                    "related_to": related.get("displayName"),
                    "related_type": related.get("type"),
                })
            return {
                # "id": attrs.get("id") or contact.get("id"),
                "display_name": attrs.get("displayName"),
                "first_name": attrs.get("firstName"),
                "last_name": attrs.get("lastName"),
                "contact_subtype": attrs.get("contactSubtype"),
                "contact_prohibited": attrs.get("contactProhibited"),
                "roles": roles,
            }

        cleaned_contacts = [clean_contact(c) for c in raw.get("data", [])]

        return {
            "count": len(cleaned_contacts),
            "contacts": cleaned_contacts
        }

    @tool
    def get_claim_vehicle_incidents(claim_id: str) -> Union[Dict[str, Any], str]:
        """
        Fetch vehicle incidents associated with a claim from a Guidewire API.

        Args:
            claim_id: Claim identifier (e.g., claim ID).

        Returns:
            Parsed JSON response (dict) when the API returns JSON; otherwise a string.
        """
        safe_claim = urllib.parse.quote(str(claim_id), safe="%")
        raw = gw.get_json(f"rest/claim/v1/claims/{safe_claim}/vehicle-incidents")
        if not isinstance(raw, dict) or "data" not in raw or not isinstance(raw["data"], list):
            return raw

        def clean_vehicle_incident(incident):
            attrs = incident.get("attributes", {})
            vehicle = attrs.get("vehicle") or {}
            driver = attrs.get("driver") or {}
            return {
                # "id": attrs.get("id") or incident.get("id"),
                "driver": driver.get("displayName"),
                "vehicle_display_name": vehicle.get("displayName"),
                "vehicle_make": vehicle.get("make"),
                "vehicle_model": vehicle.get("model"),
                "vehicle_year": vehicle.get("year"),
                "policy_vehicle": vehicle.get("policyVehicle"),
            }

        cleaned_incidents = [clean_vehicle_incident(i) for i in raw.get("data", [])]

        return {
            "count": len(cleaned_incidents),
            "vehicle_incidents": cleaned_incidents
        }

    @tool
    def get_claim_reserves(claim_id: str) -> Union[Dict[str, Any], str]:
        """
        Fetch reserves associated with a claim from a Guidewire API.

        Args:
            claim_id: Claim identifier (e.g., claim ID).

        Returns:
            Parsed JSON response (dict) when the API returns JSON; otherwise a string.
        """
        safe_claim = urllib.parse.quote(str(claim_id), safe="%")
        raw = gw.get_json(f"rest/claim/v1/claims/{safe_claim}/reserves")
        if not isinstance(raw, dict) or "data" not in raw or not isinstance(raw["data"], list):
            return raw

        def clean_reserve(reserve):
            attrs = reserve.get("attributes", {})
            reserve_line = attrs.get("reserveLine") or {}
            line_items = attrs.get("lineItems") or []
            amounts = []
            for li in line_items:
                claim_amt = li.get("claimAmount") or {}
                amounts.append({
                    "amount": claim_amt.get("amount"),
                    "currency": claim_amt.get("currency"),
                })
            return {
                "status": (attrs.get("status") or {}).get("name"),
                "subtype": (attrs.get("subtype") or {}).get("name"),
                "coverage": (attrs.get("coverage") or {}).get("name"),
                "cost_category": (reserve_line.get("costCategory") or {}).get("name"),
                "cost_type": (reserve_line.get("costType") or {}).get("name"),
                "exposure": (reserve_line.get("exposure") or {}).get("displayName"),
                "amounts": amounts,
                # "currency": (attrs.get("currency") or {}).get("name"),
                "create_time": attrs.get("createTime"),
                "created_via": (attrs.get("createdVia") or {}).get("name"),
            }

        cleaned_reserves = [clean_reserve(r) for r in raw.get("data", [])]

        return {
            "count": len(cleaned_reserves),
            "reserves": cleaned_reserves
        }

    @tool
    def get_claim_payments(claim_id: str) -> Union[Dict[str, Any], str]:
        """
        Fetch payments associated with a claim from a Guidewire API.

        Args:
            claim_id: Claim identifier (e.g., claim ID).

        Returns:
            Parsed JSON response (dict) when the API returns JSON; otherwise a string.
        """
        safe_claim = urllib.parse.quote(str(claim_id), safe="%")
        raw = gw.get_json(f"rest/claim/v1/claims/{safe_claim}/payments")
        if not isinstance(raw, dict) or "data" not in raw or not isinstance(raw["data"], list):
            return raw

        def clean_payment(payment):
            attrs = payment.get("attributes", {})
            reserve_line = attrs.get("reserveLine") or {}
            line_items = attrs.get("lineItems") or []
            amounts = []
            for li in line_items:
                claim_amt = li.get("claimAmount") or {}
                amounts.append({
                    "amount": claim_amt.get("amount"),
                    "currency": claim_amt.get("currency"),
                })
            return {
                "status": (attrs.get("status") or {}).get("name"),
                "subtype": (attrs.get("subtype") or {}).get("name"),
                "payment_type": (attrs.get("paymentType") or {}).get("name"),
                "coverage": (attrs.get("coverage") or {}).get("name"),
                "cost_category": (reserve_line.get("costCategory") or {}).get("name"),
                "cost_type": (reserve_line.get("costType") or {}).get("name"),
                "exposure": (reserve_line.get("exposure") or {}).get("displayName"),
                "amounts": amounts,
                # "currency": (attrs.get("currency") or {}).get("name"),
                "eroding": attrs.get("eroding"),
                "create_time": attrs.get("createTime"),
                "created_via": (attrs.get("createdVia") or {}).get("name"),
            }

        cleaned_payments = [clean_payment(p) for p in raw.get("data", [])]

        return {
            "count": len(cleaned_payments),
            "payments": cleaned_payments
        }

    @tool
    def get_claim_checks(claim_id: str) -> Union[Dict[str, Any], str]:
        """
        Fetch checks (disbursements) associated with a claim from a Guidewire API.

        Args:
            claim_id: Claim identifier (e.g., claim ID).

        Returns:
            Parsed JSON response (dict) when the API returns JSON; otherwise a string.
        """
        safe_claim = urllib.parse.quote(str(claim_id), safe="%")
        raw = gw.get_json(f"rest/claim/v1/claims/{safe_claim}/checks")
        if not isinstance(raw, dict) or "data" not in raw or not isinstance(raw["data"], list):
            return raw

        def clean_check(check):
            attrs = check.get("attributes", {})
            gross = attrs.get("grossAmount") or {}
            address = attrs.get("mailingAddress") or {}
            payees = []
            for p in (attrs.get("payees") or []):
                contact = p.get("contact") or {}
                payees.append({
                    "name": contact.get("displayName"),
                    "payee_type": (p.get("payeeType") or {}).get("name"),
                })
            return {
                "status": (attrs.get("status") or {}).get("name"),
                "gross_amount": gross.get("amount"),
                "currency": gross.get("currency"),
                "pay_to": attrs.get("payTo"),
                "mail_to": attrs.get("mailTo"),
                "mailing_address": address.get("displayName"),
                "payees": payees,
                "scheduled_send_date": attrs.get("scheduledSendDate"),
            }

        cleaned_checks = [clean_check(c) for c in raw.get("data", [])]

        return {
            "count": len(cleaned_checks),
            "checks": cleaned_checks
        }

    @tool
    def search_claim_documents(claim_id: str, pattern: str) -> Dict[str, Any]:
        """
        Fetch all claim documents associated with a claim from a Guidewire API, extract text from each PDF document,
        and search for a pattern in the extracted text.
        Args:
            claim_id: Claim identifier (e.g., claim ID).
            pattern: The pattern to search for in the extracted text of each PDF document.
        Returns:
            A dictionary containing the claim ID, and a list of PDF documents with their extracted text and grep results.
        """
        safe_claim = urllib.parse.quote(str(claim_id), safe="%")
        documents_response = gw.get_json(f"rest/claim/v1/claims/{safe_claim}/documents")
 
        if not isinstance(documents_response, dict):
            return {
                "claim_id": claim_id,
                "error": "unexpected_documents_response",
                "documents_response": documents_response,
            }
 
        if documents_response.get("error"):
            return documents_response
 
        pdf_documents = []
 
        for doc in documents_response.get("data", []):
            attributes = doc.get("attributes", {})
            mime_type = attributes.get("mimeType")
            document_id = attributes.get("id")
 
            if mime_type == "application/pdf" and document_id:
                document_result = gw.get_document_text_content(claim_id, document_id)
                extracted_text = document_result["document_text"]
 
                grep_result = None
                if extracted_text:
                    grep_result = gw.grep_file(pattern, extracted_text)
 
                pdf_documents.append({
                    # "document_id": document_id,
                    "name": attributes.get("name"),
                    # "document_text": document_result["document_text"],
                    "grep_result": grep_result,
                    # "fetch_status": document_result["fetch_status"],
                    # "content_error": document_result["content_error"],
                })
 
        results_str, truncated = gw.truncate_to_token_limit(pdf_documents, max_tokens=4000)
 
        return {
            "claim_id": claim_id,
            "count": len(pdf_documents),
            "results": results_str,
            # "truncated": truncated,
        }
 
    return {
        "get_claim_details": get_claim_details,
        "get_claim_activities": get_claim_activities,
        "get_claim_notes": get_claim_notes,
        "get_claim_exposures": get_claim_exposures,
        "get_claim_contacts": get_claim_contacts,
        "get_claim_vehicle_incidents": get_claim_vehicle_incidents,
        "get_claim_reserves": get_claim_reserves,
        "get_claim_payments": get_claim_payments,
        "get_claim_checks": get_claim_checks,
        "get_claim_history_events": get_claim_history_events,
        "search_claim_documents": search_claim_documents,
        "get_policy_details": get_policy_details,
    }
 
 
if __name__ == "__main__":
    tools = make_guidewire_tools(
        "https://cc-dev-gwcpdev.valuemom.zeta1-andromeda.guidewire.net:443",
        username="su",
        password="gw",
    )
 
    claim_id = "**********"
 
    tool_names = [
        "get_claim_details",
        "get_claim_activities",
        "get_claim_notes",
        "get_claim_exposures",
        "get_claim_contacts",
        "get_claim_vehicle_incidents",
        "get_claim_reserves",
        "get_claim_payments",
        "get_claim_checks",
        "get_claim_history_events",
        "search_claim_documents",
        "get_policy_details",
    ]
 
    for name in tool_names:
        print(f"================== {name} ==============================")
        print(tools[name].invoke({"claim_id": claim_id, "pattern": "Latin"}))