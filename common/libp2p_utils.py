import logging
from nacl.signing import SigningKey
from libp2p.crypto.keys import PublicKey, PrivateKey, KeyPair, KeyType
from libp2p.peer.id import ID as PeerID

logger = logging.getLogger(__name__)

# --- Workaround for py-libp2p Ed25519 key generation ---
# Until py-libp2p offers its own factory, we use PyNaCl and wrap it.

class _ED25519Pub(PublicKey):
    """Internal wrapper for Ed25519 Public Key from PyNaCl."""
    def __init__(self, vk_bytes: bytes):
        self._vk = vk_bytes

    def to_bytes(self) -> bytes:
        return self._vk

    def get_type(self) -> KeyType:
        return KeyType.Ed25519

    def verify(self, data: bytes, sig: bytes) -> bool:
        """Verifies a signature using the PyNaCl VerifyKey."""
        from nacl.signing import VerifyKey # Import locally to avoid circular deps if not needed elsewhere
        try:
            VerifyKey(self._vk).verify(data, sig)
            return True
        except Exception: # nacl.exceptions.BadSignatureError
            return False

class _ED25519Priv(PrivateKey):
    """Internal wrapper for Ed25519 Private Key (SigningKey) from PyNaCl."""
    def __init__(self, sk: SigningKey):
        self._sk = sk

    def to_bytes(self) -> bytes:
        return self._sk.encode()

    def get_type(self) -> KeyType:
        return KeyType.Ed25519

    def sign(self, data: bytes) -> bytes:
        """Signs data using the PyNaCl SigningKey."""
        return self._sk.sign(data).signature

    def get_public_key(self) -> PublicKey:
        """Returns the corresponding public key."""
        return _ED25519Pub(self._sk.verify_key.encode())

    @property
    def public_key(self) -> PublicKey:
        """Returns the corresponding public key as a property."""
        return self.get_public_key()

    @property # ADD THIS DECORATOR
    def private_key(self) -> PrivateKey: # ADD THIS PROPERTY
        """Returns the private key itself as a property."""
        return self # The NoiseTransport expects the PrivateKey object itself

def generate_libp2p_keypair() -> KeyPair:
    """
    Generates an Ed25519 keypair using PyNaCl and wraps it into libp2p's KeyPair object.
    This serves as a workaround until py-libp2p provides its own key generation.

    Returns:
        KeyPair: A libp2p KeyPair object.
    """
    logger.info("Generating libp2p Ed25519 keypair using PyNaCl workaround.")
    signing_key = SigningKey.generate()
    priv = _ED25519Priv(signing_key)
    pub = priv.get_public_key()
    key_pair = KeyPair(private_key=priv, public_key=pub)
    peer_id = PeerID.from_pubkey(key_pair.public_key)
    logger.info(f"Generated libp2p KeyPair with PeerID: {peer_id}")
    return key_pair
